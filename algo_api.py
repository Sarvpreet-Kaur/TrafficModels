from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from algorithm import DynamicTrafficController

app = FastAPI()

# ------------------ CORS------------------ #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # allow dashboard running locally
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

controller = None
current_lanes = []

class LaneInput(BaseModel):
    lane_id: str
    normal: int
    emergency: int


@app.post("/update")
def update_signal(input_data: List[LaneInput]):
    global controller, current_lanes

    lane_list = [item.dict() for item in input_data]
    incoming_lanes = [lane["lane_id"] for lane in lane_list]

    if controller is None or set(incoming_lanes) != set(current_lanes):
        controller = DynamicTrafficController(
            N=len(incoming_lanes),
            min_green=3.0,
            max_green=12.0,
            yellow_time=2.0,
            clearance_rate=2.5,
            debug=False
        )

        controller.lane_ids = incoming_lanes

        controller.lanes = {
            lane_id: {"normal": 0, "emergency": 0, "wait": 0, "state": [1, 0, 0]}
            for lane_id in incoming_lanes
        }

        current_lanes = incoming_lanes

    result = controller.update(lane_list)

    return {"status": "success", "output": result}


@app.get("/status")
def get_status():
    return controller.lanes if controller else {}
