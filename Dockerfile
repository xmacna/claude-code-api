# Use Ubuntu as base for better Claude Code support
FROM ubuntu:22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Node.js and Python 3.11
RUN apt-get update && apt-get install -y \
    curl \
    git \
    software-properties-common \
    ca-certificates \
    bash \
    sudo \
    jq \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-venv python3.11-dev \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools to latest versions
RUN pip3 install --upgrade pip setuptools wheel

# Install Node.js 18+ (required for Claude Code)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    node --version && npm --version

# Install Claude Code CLI globally as root (before switching to non-root user)
RUN npm install -g @anthropic-ai/claude-code && \
    claude --version

# Create non-root user for running Claude Code
RUN useradd -m -s /bin/bash claudeuser && \
    echo "claudeuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to non-root user
USER claudeuser
WORKDIR /home/claudeuser

# Create Claude config directory
RUN mkdir -p /home/claudeuser/.config/claude

# Create workspace directory for Claude Code
RUN mkdir -p /home/claudeuser/workspace

# Set up working directory
WORKDIR /home/claudeuser/app

# Clone claude-code-api
RUN git clone https://github.com/xmacna/claude-code-api.git .

# Install dependencies using modern pip (avoiding deprecated setup.py)
RUN pip3 install --user --upgrade pip && \
    pip3 install --user -e . --use-pep517 || \
    pip3 install --user -e .

# Add user's local bin to PATH
ENV PATH="/home/claudeuser/.local/bin:${PATH}"

# Expose API port
EXPOSE 8000

# Environment variables (set these at runtime)
ENV ANTHROPIC_API_KEY=""
ENV HOST=0.0.0.0
ENV PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Create entrypoint script - DO NOT configure with API key if using Claude Max
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Only configure API key if explicitly provided and not using Claude Max\n\
if [ -n "$ANTHROPIC_API_KEY" ] && [ "$USE_CLAUDE_MAX" != "true" ]; then\n\
  echo "Configuring Claude Code with API key..."\n\
  mkdir -p ~/.config/claude\n\
  cat > ~/.config/claude/config.json << EOF\n\
{\n\
  "apiKey": "$ANTHROPIC_API_KEY",\n\
  "autoUpdate": false\n\
}\n\
EOF\n\
  echo "Claude Code configured with API key"\n\
elif [ "$USE_CLAUDE_MAX" = "true" ]; then\n\
  echo "Using Claude Max subscription - please run: docker exec -it claude-code-api claude"\n\
  echo "Then authenticate via browser when prompted"\n\
else\n\
  echo "No authentication configured. Set ANTHROPIC_API_KEY or USE_CLAUDE_MAX=true"\n\
fi\n\
\n\
# Test Claude Code\n\
echo "Testing Claude Code..."\n\
claude --version || echo "Claude Code installed"\n\
\n\
echo "Starting API server..."\n\
cd /home/claudeuser/app\n\
exec python3 -m claude_code_api.main' > /home/claudeuser/entrypoint.sh && \
    chmod +x /home/claudeuser/entrypoint.sh

# Start the API server with entrypoint
ENTRYPOINT ["/home/claudeuser/entrypoint.sh"]