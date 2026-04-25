"""
FastAPI application for the Afaa Environment.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required for the web interface.") from e

try:
    # Standard package-style imports
    from ..models import AfaaAction, AfaaObservation
    from .AFAA_environment import AfaaEnvironment
except (ModuleNotFoundError, ImportError):
    # Fallback for local running
    from models import AfaaAction, AfaaObservation
    from server.AFAA_environment import AfaaEnvironment

# Create the application instance
app = create_app(
    AfaaEnvironment,
    AfaaAction,
    AfaaObservation,
    env_name="AFAA",
    max_concurrent_envs=5,
)

def main():
    """
    Entry point for the openenv CLI and project scripts.
    """
    import uvicorn
    import argparse
    import os

    # Use environment variables or arguments for port/host
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8000)))
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()