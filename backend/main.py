import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import api_router

def main():
    app = FastAPI(title="Medical Fake News Detector API")

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router=api_router)
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()