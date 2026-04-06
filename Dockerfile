# ============================================================================
# OpenARC From Scratch - Ubuntu Base + Manual Intel Setup
# ============================================================================
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# ============================================================================
# System Dependencies
# ============================================================================
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    git \
    gpg \
    gpg-agent \
    libdrm2 \
    libnuma1 \
    libudev1 \
    ocl-icd-libopencl1 \
    wget \
    python3 \
    python3-venv \
    python3-dev \
    python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    rm -rf /var/lib/apt/lists/*

# ============================================================================
# Intel GPU Drivers
# ============================================================================
RUN wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
    gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble client" | \
    tee /etc/apt/sources.list.d/intel-gpu-noble.list && \
    apt-get update && apt-get install -y \
    clinfo \
    intel-opencl-icd \
    intel-level-zero-gpu \
    libze1 \
    level-zero \
    level-zero-dev && \
    ldconfig && \
    test -s /etc/OpenCL/vendors/intel.icd && \
    test -f /usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so && \
    ldd /usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so | tee /tmp/ldd-igdrcl.txt && \
    ! grep -qi "not found" /tmp/ldd-igdrcl.txt && \
    OCL_ICD_DEBUG=7 clinfo >/tmp/clinfo.log 2>&1 && \
    grep -q "Platform Name" /tmp/clinfo.log && \
    rm -rf /var/lib/apt/lists/*

# ============================================================================
# Intel NPU Driver
# ============================================================================
# NOTE:
# Keep NPU runtime out of the default image to avoid /usr/local linker-path
# collisions with Intel GPU OpenCL/Level Zero userspace.
# Build/install the NPU stack in a dedicated image when needed.

# ============================================================================
# Install uv package manager
# ============================================================================
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# ============================================================================
# Copy and setup OpenArc
# ============================================================================
WORKDIR /app
COPY . /app
RUN echo "OpenARC version: $(git describe --tags --always || echo 'local-checkout')"

# ============================================================================
# Install Python dependencies with uv
# ============================================================================
RUN uv sync && \
    uv pip install "optimum-intel[openvino] @ git+https://github.com/huggingface/optimum-intel" && \
    uv pip install --pre -U openvino-genai openvino-tokenizers \
        --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly

# Add venv to PATH so openarc command works
ENV PATH="/app/.venv/bin:$PATH"

# ============================================================================
# Runtime Configuration
# ============================================================================
ENV NEOReadDebugKeys=1 \
    OverrideGpuAddressSpace=48 \
    EnableImplicitScaling=1 \
    OPENARC_API_KEY=key \
    OPENARC_AUTOLOAD_MODEL=""

# Create persistent config directory and symlink
RUN mkdir -p /persist && \
    ln -sf /persist/openarc_config.json /app/openarc_config.json

# ============================================================================
# Build Info Logging
# ============================================================================
RUN echo "=== Build Information ===" > /app/BUILD_INFO.txt && \
    echo "Build Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")" >> /app/BUILD_INFO.txt && \
    echo "OpenARC Version: $(git describe --tags --always)" >> /app/BUILD_INFO.txt && \
    echo "" >> /app/BUILD_INFO.txt && \
    echo "=== Intel Package Versions ===" >> /app/BUILD_INFO.txt && \
    uv pip list | grep -E "(openvino|optimum|torch)" >> /app/BUILD_INFO.txt || true && \
    echo "" >> /app/BUILD_INFO.txt && \
    echo "=== System Package Versions ===" >> /app/BUILD_INFO.txt && \
    dpkg -l | grep -E "intel-opencl|level-zero" | awk '{print $2 " " $3}' >> /app/BUILD_INFO.txt || true

# ============================================================================
# Startup Script
# ============================================================================
RUN cat > /usr/local/bin/start-openarc.sh <<'SCRIPT'
#!/bin/bash
set -e

echo "================================================"
echo "=== Starting OpenArc Server ==="
echo "================================================"

if [ -f /app/BUILD_INFO.txt ]; then
  cat /app/BUILD_INFO.txt
  echo ""
fi

echo "=== Runtime Configuration ==="
echo "Port: 8000"
echo "API Key: ${OPENARC_API_KEY:0:10}..."
echo "Auto-load Model: ${OPENARC_AUTOLOAD_MODEL:-none}"
echo "OpenVINO devices: $(python3 -c 'from openvino import Core; print(Core().available_devices)' 2>/tmp/ov_device_err.log || cat /tmp/ov_device_err.log)"
echo "OpenCL ICDs:"
ls -1 /etc/OpenCL/vendors 2>/dev/null || echo "(none)"
echo "/dev/dri:"
ls -l /dev/dri 2>/dev/null || echo "(missing)"
echo ""
echo "================================================"

# Start server in background
openarc serve start --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

# Auto-load model if specified
if [ -n "$OPENARC_AUTOLOAD_MODEL" ]; then
  echo "Waiting for server to start..."
  for i in {1..30}; do
    if curl -s -f -H "Authorization: Bearer ${OPENARC_API_KEY}" http://localhost:8000/v1/models >/dev/null 2>&1; then
      echo "Server ready after $i seconds"
      echo "Auto-loading model: $OPENARC_AUTOLOAD_MODEL"
      openarc load "$OPENARC_AUTOLOAD_MODEL" || echo "Failed to auto-load model"
      break
    fi
    sleep 1
  done
fi

# Wait for server
wait $SERVER_PID
SCRIPT

RUN chmod +x /usr/local/bin/start-openarc.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f -H "Authorization: Bearer ${OPENARC_API_KEY}" http://localhost:8000/v1/models || exit 1

CMD ["/usr/local/bin/start-openarc.sh"]
