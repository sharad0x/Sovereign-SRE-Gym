"""
FastAPI application for the Afaa Environment.
"""

# server/app.py
try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required for the web interface.") from e

try:
    from ..models import AfaaAction, AfaaObservation
    from .AFAA_environment import AfaaEnvironment
except (ModuleNotFoundError, ImportError):
    from models import AfaaAction, AfaaObservation
    from server.AFAA_environment import AfaaEnvironment

# FIXED: Removed AfaaState from positional arguments
app = create_app(
    AfaaEnvironment,
    AfaaAction,
    AfaaObservation,
    env_name="AFAA",
    max_concurrent_envs=5,
)

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)