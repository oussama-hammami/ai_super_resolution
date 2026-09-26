import glfw
from OpenGL.GL import *
from cuda.bindings import runtime as cudart


def _check(result, name):
    """cuda-python returns (err, *values); raise on error, return the values."""
    err, *values = result
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{name}: {err}")
    return values[0] if len(values) == 1 else values


class GLDisplay:
    """
    Persistent window that displays [H, W, 3] uint8 RGB CUDA tensors via
    CUDA-GL interop. Everything (window, PBO, texture, CUDA registration) is
    created once; each show() is a device-to-device copy into the Pixel Buffer
    Object (PBO), a GPU-side PBO -> texture upload and one textured quad.
    The frame never leaves the GPU.

    All methods must be called from the thread that created the instance,
    since that thread owns the OpenGL context.
    """

    def __init__(self, H, W, title="Super Resolution", vsync=True):
        self.H, self.W = H, W
        self.nbytes = H * W * 3

        # ------------------------------------------------------------------
        # 1. GLFW + OpenGL context, window scaled to fit the screen
        # ------------------------------------------------------------------
        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        _, _, screen_w, screen_h = glfw.get_monitor_workarea(glfw.get_primary_monitor())
        if screen_w == 0 or screen_h == 0:
            mode = glfw.get_video_mode(glfw.get_primary_monitor())
            screen_w, screen_h = mode.size.width, mode.size.height
        fit = min(1.0, 0.9 * screen_w / W, 0.9 * screen_h / H)

        self.window = glfw.create_window(int(W * fit), int(H * fit), title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")
        glfw.set_window_aspect_ratio(self.window, W, H)

        glfw.make_context_current(self.window)
        glfw.swap_interval(1 if vsync else 0)

        # ------------------------------------------------------------------
        # 2. Pixel Buffer Object (PBO), registered with CUDA (write-discard:
        #    we only write from CUDA)
        # ------------------------------------------------------------------
        self.pbo = glGenBuffers(1)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo)
        glBufferData(GL_PIXEL_UNPACK_BUFFER, self.nbytes, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        self.resource = _check(cudart.cudaGraphicsGLRegisterBuffer(
            int(self.pbo),
            cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
        ), "cudaGraphicsGLRegisterBuffer")

        # ------------------------------------------------------------------
        # 3. Texture the PBO is uploaded into; RGB rows are not 4-byte aligned
        #    in general, so unpack with 1-byte alignment
        # ------------------------------------------------------------------
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        self.tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, W, H, 0, GL_RGB, GL_UNSIGNED_BYTE, None)

        # ------------------------------------------------------------------
        # 4. Unit-square projection, top-left origin matches image convention
        # ------------------------------------------------------------------
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, 1, 1, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glEnable(GL_TEXTURE_2D)

    def show(self, img):
        """Display a contiguous [H, W, 3] uint8 RGB CUDA tensor."""
        assert img.is_cuda and img.dtype.itemsize == 1 and img.is_contiguous()
        assert tuple(img.shape) == (self.H, self.W, 3)

        # map -> GPU memcpy -> unmap (zero CPU involvement)
        _check(cudart.cudaGraphicsMapResources(1, self.resource, 0), "cudaGraphicsMapResources")
        dev_ptr, _ = _check(cudart.cudaGraphicsResourceGetMappedPointer(self.resource),
                            "cudaGraphicsResourceGetMappedPointer")
        _check(cudart.cudaMemcpy(dev_ptr, img.data_ptr(), self.nbytes,
                                 cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice), "cudaMemcpy")
        _check(cudart.cudaGraphicsUnmapResources(1, self.resource, 0), "cudaGraphicsUnmapResources")

        # PBO -> texture (GPU DMA path, no CPU read-back)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.W, self.H, GL_RGB, GL_UNSIGNED_BYTE, None)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        # fullscreen textured quad, scaled to the current framebuffer size
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        glViewport(0, 0, fb_w, fb_h)
        glClear(GL_COLOR_BUFFER_BIT)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(1, 0)
        glTexCoord2f(1, 1); glVertex2f(1, 1)
        glTexCoord2f(0, 1); glVertex2f(0, 1)
        glEnd()
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def should_close(self):
        return (glfw.window_should_close(self.window)
                or glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS)

    def set_title(self, title):
        glfw.set_window_title(self.window, title)

    def close(self):
        cudart.cudaGraphicsUnregisterResource(self.resource)
        glDeleteBuffers(1, [self.pbo])
        glDeleteTextures(1, [self.tex])
        glfw.destroy_window(self.window)
        glfw.terminate()


def display_image_gl(tensor):
    """
    Display a [3, H, W] uint8 CUDA tensor until the window is closed.
    The tensor never leaves the GPU.
    """
    assert tensor.is_cuda, "tensor must reside on a CUDA device"
    assert tensor.dtype.itemsize == 1, "tensor must be uint8"

    # CHW -> HWC, contiguous so data_ptr() is a flat RGB buffer
    img = tensor.permute(1, 2, 0).contiguous()
    display = GLDisplay(img.shape[0], img.shape[1])
    while not display.should_close():
        display.show(img)
    display.close()
