from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch

# Initialize the FastAPI app
app = FastAPI()

# CORS setup
origins = [
    "http://localhost",  # Allow requests from localhost
    "http://localhost:8000",  # Specific port
    "*"
]

# Add CORS middleware to allow cross-origin requests from the specified origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow requests from specific origins
    allow_credentials=True,  # Enable credentials (e.g., cookies, authentication)
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Model paths - update these paths with your actual model file locations
MODEL_TYPE_PATH = "models/lem/LEMv3_MODEL_TYPE_TORCH.pth"
MODEL_ACC_PATH = "models/lem/LEMv4_MODEL_ACC_TORCH.pth"
MODEL_DATA_PATH = "models/lem/LEMv3_MODEL_DATA_TORCH.pth"

# Load pre-trained models
def load_model(model_path):
    try:
        model = torch.load(model_path)
        model.eval()  # Set to evaluation mode
        return model
    except Exception as e:
        raise Exception(f"Error loading model {model_path}: {str(e)}")

model_type = load_model(MODEL_TYPE_PATH)
model_acc = load_model(MODEL_ACC_PATH)
model_data = load_model(MODEL_DATA_PATH)

# Input schema for prediction requests
class EventInput(BaseModel):
    event_type: int      # One-hot encoded or categorical event type
    period: int          # Period of the game (0: first half, 1: second half)
    minute: float        # Time elapsed in the game normalized 0-1
    x: float             # X coordinate (normalized)
    y: float             # Y coordinate (normalized)
    is_home_team: bool   # Is home team performing the event
    home_score: float    # Home team's score normalized 0-1
    away_score: float    # Away team's score normalized 0-1

# Match state to keep track of scores
match_state = {
    "home_score": 0,
    "away_score": 0
}

# Prediction logic
def predict_next_event(input_data: EventInput):
    global match_state  # Access match state

    # Update scores if the event is a goal (assuming event_type == 1 is a goal)
    if input_data.event_type == 1:  # Assuming event_type 1 is a goal
        if input_data.is_home_team:
            match_state["home_score"] += 1
        else:
            match_state["away_score"] += 1

    # Original input tensor (8 features)
    input_tensor = torch.tensor([
        input_data.event_type, input_data.period, input_data.minute,
        input_data.x, input_data.y, input_data.is_home_team, input_data.home_score, input_data.away_score
    ], dtype=torch.float32)

    # Pad input tensor to match expected input sizes
    input_type = torch.cat([input_tensor, torch.zeros(42 - 8)])  # Pad to 42 features for model_type
    input_acc = torch.cat([input_tensor, torch.zeros(75 - 8)])   # Pad to 75 features for model_acc
    input_data_padded = torch.cat([input_tensor, torch.zeros(77 - 8)])  # Pad to 77 features for model_data

    # Predict event type
    with torch.no_grad():
        event_type_pred = model_type(input_type)

    # Predict accuracy and event data
    with torch.no_grad():
        event_acc_pred = model_acc(input_acc)
        event_data_pred = model_data(input_data_padded)

    # Handle event_acc_pred output (which likely has 2 elements)
    event_acc_pred_sigmoid = torch.sigmoid(event_acc_pred)

    # We'll return both values of event_acc_pred (you can adjust as needed)
    predicted_accuracy = event_acc_pred_sigmoid.mean().item()

    # Calculate match result and goal difference
    home_score = match_state["home_score"]
    away_score = match_state["away_score"]

    if home_score > away_score:
        result = "Home Win"
        goal_diff = home_score - away_score
    elif home_score < away_score:
        result = "Away Win"
        goal_diff = away_score - home_score
    else:
        result = "Draw"
        goal_diff = 0

    # Prepare the output
    output = {
        "predicted_event_type": event_type_pred.argmax().item(),
        "predicted_accuracy": predicted_accuracy,  # Use mean or select first value
        "predicted_x": event_data_pred[0].item(),
        "predicted_y": event_data_pred[1].item(),
        "predicted_time": event_data_pred[2].item(),
        "home_score": home_score,
        "away_score": away_score,
        "match_result": result,
        "goal_difference": goal_diff
    }
    return output

# API route for predictions
@app.post("/predict")
async def predict_event(input_data: EventInput):
    try:
        prediction = predict_next_event(input_data)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# Root endpoint to verify the API is running
@app.get("/")
def read_root():
    return {"message": "LEM FastAPI with CORS is running"}