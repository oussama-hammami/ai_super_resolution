import glfw
from OpenGL.GL import *
from cuda import cudart


def display_image_gl(tensor):
    """
    Display a [3, H, W] uint8 CUDA tensor via OpenGL using CUDA-GL interop.
    The tensor never leaves the GPU — it is copied directly into an OpenGL
    Pixel Buffer Object (PBO) on the device, then DMA'd into a texture.
    """
    assert tensor.is_cuda, "tensor must reside on a CUDA device"
    assert tensor.dtype.itemsize == 1, "tensor must be uint8"

    # CHW → HWC, contiguous so data_ptr() is a flat RGB buffer
    img = tensor.permute(1, 2, 0).contiguous()
    _, H, W = tensor.shape
    nbytes = H * W * 3

    # ------------------------------------------------------------------
    # 1. GLFW + OpenGL context
    # ------------------------------------------------------------------
    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
    window = glfw.create_window(W, H, "Super Resolution", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW window creation failed")

    glfw.make_context_current(window)
    glfw.swap_interval(1)   # vsync

    # ------------------------------------------------------------------
    # 2. Pixel Buffer Object (PBO)
    # ------------------------------------------------------------------
    pbo = glGenBuffers(1)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo)
    glBufferData(GL_PIXEL_UNPACK_BUFFER, nbytes, None, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    # ------------------------------------------------------------------
    # 3. Register PBO with CUDA (write-discard: we only write from CUDA)
    # ------------------------------------------------------------------
    err, resource = cudart.cudaGraphicsGLRegisterBuffer(
        int(pbo),
        cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
    )
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaGraphicsGLRegisterBuffer: {err}")

    # ------------------------------------------------------------------
    # 4. Map → GPU memcpy → unmap  (zero CPU involvement)
    # ------------------------------------------------------------------
    (err,) = cudart.cudaGraphicsMapResources(1, resource, 0)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaGraphicsMapResources: {err}")

    err, dev_ptr, _ = cudart.cudaGraphicsResourceGetMappedPointer(resource)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaGraphicsResourceGetMappedPointer: {err}")

    (err,) = cudart.cudaMemcpy(
        dev_ptr, img.data_ptr(), nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
    )
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaMemcpy: {err}")

    (err,) = cudart.cudaGraphicsUnmapResources(1, resource, 0)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaGraphicsUnmapResources: {err}")

    # ------------------------------------------------------------------
    # 5. Upload PBO → texture (GPU DMA path, no CPU read-back)
    # ------------------------------------------------------------------
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, W, H, 0, GL_RGB, GL_UNSIGNED_BYTE, None)

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, None)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    # ------------------------------------------------------------------
    # 6. Render loop — fullscreen textured quad, fixed-function pipeline
    # ------------------------------------------------------------------
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, W, H, 0, -1, 1)   # top-left origin matches image convention
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glEnable(GL_TEXTURE_2D)

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)
        glBindTexture(GL_TEXTURE_2D, tex)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(W, 0)
        glTexCoord2f(1, 1); glVertex2f(W, H)
        glTexCoord2f(0, 1); glVertex2f(0, H)
        glEnd()
        glfw.swap_buffers(window)
        glfw.poll_events()

    # ------------------------------------------------------------------
    # 7. Cleanup
    # ------------------------------------------------------------------
    cudart.cudaGraphicsUnregisterResource(resource)
    glDeleteBuffers(1, [pbo])
    glDeleteTextures(1, [tex])
    glfw.destroy_window(window)
    glfw.terminate()
