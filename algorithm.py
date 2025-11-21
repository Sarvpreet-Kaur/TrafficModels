import time
import random
from typing import List, Dict, Any

class DynamicTrafficController:
    """
    Dynamic traffic signal controller that decides which lane to open
    based on:
      - Number of emergency and normal vehicles
      - Lane waiting time (aging/fairness)
      - Dynamic green time proportional to traffic volume
    """

    def __init__(
        self,
        N: int,
        yellow_time: float = 2.0,
        min_green: float = 3.0,
        max_green: float = 15.0,
        wait_boost: float = 0.4,
        starvation_limit: int = 8,
        clearance_rate: float = 3.0,
        debug: bool = True
    ):
        # Initialize lane data
        self.lane_ids = [f"Lane_{i+1}" for i in range(N)]
        self.lanes = {
            lane: {"normal": 0, "emergency": 0, "wait": 0, "state": [1, 0, 0]}
            for lane in self.lane_ids
        }

        # State tracking
        self.current_green = None
        self.green_started_at = None
        self.current_green_time = min_green
        self.last_emergency_lane = None

        # Parameters
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.wait_boost = wait_boost
        self.starvation_limit = starvation_limit
        self.clearance_rate = clearance_rate
        self.debug = debug

    # Emergency lane chooser (highest priority)
    def _choose_emergency_lane(self, data: Dict[str, Any]):
        emergencies = [l for l in data if data[l]["emergency"] > 0]
        if not emergencies:
            return None

        # Select lane with highest emergency count
        max_count = max(data[l]["emergency"] for l in emergencies)
        tied = [l for l in emergencies if data[l]["emergency"] == max_count]

        # Round-robin tie breaker
        if len(tied) == 1:
            chosen = tied[0]
        else:
            start = 0
            if self.last_emergency_lane in self.lane_ids:
                start = (self.lane_ids.index(self.last_emergency_lane) + 1) % len(self.lane_ids)
            for i in range(len(self.lane_ids)):
                lane = self.lane_ids[(start + i) % len(self.lane_ids)]
                if lane in tied:
                    chosen = lane
                    break
            else:
                chosen = tied[0]
        self.last_emergency_lane = chosen
        return chosen

    # Normal lane chooser (vehicles + fairness)
    def _choose_normal_lane(self, data: Dict[str, Any]):
        scores = {}
        for lane in self.lane_ids:
            n = data[lane]["normal"]
            w = data[lane]["wait"]

            # Base score proportional to number of vehicles and wait boost
            score = n * (1 + w * self.wait_boost)

            # Starvation prevention
            if w >= self.starvation_limit:
                score += 1000

            scores[lane] = score

        chosen = max(scores, key=scores.get)
        return chosen

    # Dynamic green time calculation
    def _calculate_green_time(self, lane_data):
        """
        Green time proportional to vehicle count, capped between min & max.
        Emergency vehicles add extra time.
        """
        normal = lane_data["normal"]
        emergency = lane_data["emergency"]
        wait = lane_data["wait"]

        # Balanced formula (recommended)
        clear_time = normal / self.clearance_rate
        wait_bonus = wait * 0.4
        emergency_bonus = emergency * 2.0

        base_time = clear_time + wait_bonus + emergency_bonus
        green_time = max(self.min_green, min(base_time, self.max_green))
        return green_time

    # Transition phase (yellow then green)
    def _apply_yellow(self, to_green: str):
        for lane in self.lane_ids:
            if lane == to_green:
                self.lanes[lane]["state"] = [0, 1, 0]  # yellow
            else:
                self.lanes[lane]["state"] = [1, 0, 0]  # red

        # Transition instantly to green
        self.lanes[to_green]["state"] = [0, 0, 1]
        self.current_green = to_green
        self.green_started_at = time.time()

    # Update wait counters
    def _update_waits(self, chosen: str):
        for lane in self.lane_ids:
            if lane == chosen:
                self.lanes[lane]["wait"] = 0
            else:
                self.lanes[lane]["wait"] += 1

    # Simulate vehicle flow
    def _simulate_flow(self, data: Dict[str, Any], chosen: str, green_time: float):
        normal = data[chosen]["normal"]
        cleared = min(normal, int(self.clearance_rate * green_time))
        data[chosen]["normal"] -= cleared

        # New random arrivals in other lanes
        for lane in self.lane_ids:
            if lane != chosen:
                data[lane]["normal"] += random.randint(0, 3)

    # Main update function
    def update(self, lanes_data: List[Dict[str, Any]]) -> Dict[str, Any]:

        # Prepare internal data
        data = {
            d["lane_id"]: {
                "normal": d["normal"],
                "emergency": d["emergency"],
                "wait": self.lanes[d["lane_id"]]["wait"]
            }
            for d in lanes_data
        }

        # Step 1: Emergencies first
        chosen = self._choose_emergency_lane(data)

        # Step 2: Normal lane selection if no emergency
        if not chosen:
            if self.current_green and self.green_started_at:
                elapsed = time.time() - self.green_started_at
                if elapsed < self.current_green_time:
                    chosen = self.current_green
                else:
                    chosen = self._choose_normal_lane(data)
            else:
                chosen = self._choose_normal_lane(data)

        # Step 3: Update wait times
        self._update_waits(chosen)

        # Step 4: Calculate dynamic green time
        self.current_green_time = self._calculate_green_time(data[chosen])

        # Step 5: Apply yellow → green
        self._apply_yellow(chosen)

        # Step 6: Simulate flow
        self._simulate_flow(data, chosen, self.current_green_time)

        # Step 7: Update internal queues
        for lane in self.lane_ids:
            self.lanes[lane]["normal"] = data[lane]["normal"]

        # Debug output
        if self.debug:
            print("\n==============================")
            print(f" Active Green: {chosen}")
            print(f" Green Time  : {self.current_green_time:.1f}s")
            print(f" Wait Times  : {[self.lanes[l]['wait'] for l in self.lane_ids]}")
            print(" Lane States :", {l: self.lanes[l]["state"] for l in self.lane_ids})
            print("==============================\n")

        # Final enriched output (state + wait + green time)
        output = {}
        for lane in self.lane_ids:
            output[lane] = {
                "state": self.lanes[lane]["state"],
                "wait": self.lanes[lane]["wait"],
            }

        # Only current green lane has green_time
        output[self.current_green]["green_time"] = self.current_green_time

        return output


# # ===================== Simulation Example ===================== #

# # Create a controller for 4 lanes
# controller = DynamicTrafficController(
#     N=4,
#     min_green=3.0,
#     max_green=12.0,
#     yellow_time=2.0,
#     clearance_rate=2.5,
#     debug=True
# )

# # Initialize random vehicle counts
# lanes_data = [
#     {"lane_id": "Lane_1", "normal": 5, "emergency": 0},
#     {"lane_id": "Lane_2", "normal": 3, "emergency": 0},
#     {"lane_id": "Lane_3", "normal": 6, "emergency": 0},
#     {"lane_id": "Lane_4", "normal": 4, "emergency": 0},
# ]

# # Run simulation for 10 cycles
# for cycle in range(10):
#     print(f"\n======= Cycle {cycle + 1} =======")

#     # Random new vehicles + occasional emergency
#     for lane in lanes_data:
#         lane["normal"] += random.randint(0, 3)
#         lane["emergency"] = 1 if random.random() < 0.1 else 0

#     # Feed to controller
#     output = controller.update(lanes_data)

#     print("Output:", output)
#     time.sleep(1)
