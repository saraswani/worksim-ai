from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os

from env import WorkSimEnv, Action

app = FastAPI(title="WorkSim AI - OpenEnv Server")

# Global environment instance (in-memory for the hackathon)
# In a real production setup, you might want session-based envs
_env_instances: Dict[str, WorkSimEnv] = {
    "default": WorkSimEnv(task_name="email_triage")
}

class StepRequest(BaseModel):
    action: Dict[str, Any]
    task_id: Optional[str] = "default"

@app.get("/")
def health_check():
    return {"status": "ok", "message": "WorkSim AI OpenEnv server is running"}

class ResetRequest(BaseModel):
    task_id: Optional[str] = "email_triage"

@app.post("/reset")
def reset_env(request: Optional[ResetRequest] = None, task_id: Optional[str] = None):
    # Determine the task_id from body or query param
    t_id = "email_triage"
    if request and request.task_id:
        t_id = request.task_id
    elif task_id:
        t_id = task_id
        
    # If a new task_id is provide, re-init the env
    if t_id not in _env_instances:
        try:
            _env_instances["default"] = WorkSimEnv(task_name=t_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    _env_instances["default"].task_id = t_id # Ensure the instance is set to the right task
    obs = _env_instances["default"].reset()
    return obs.dict()

@app.post("/step")
def step_env(request: StepRequest):
    env = _env_instances.get("default")
    if not env:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    try:
        # Map the input dict to the Action Pydantic model
        action_obj = Action(**request.action)
        obs, reward, done, info = env.step(action_obj)
        
        return {
            "observation": obs.dict(),
            "reward": reward.dict(),
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def get_state():
    env = _env_instances.get("default")
    if not env:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    return env.state()

if __name__ == "__main__":
    import uvicorn
    # Use port 7860 as it is standard for Hugging Face Spaces
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
