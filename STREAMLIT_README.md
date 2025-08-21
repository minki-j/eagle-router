# Eagle Router Streamlit Demo

This is a Streamlit application that demonstrates the Eagle Router's intelligent model selection capabilities with user feedback.

## Features

- **Interactive Model Routing**: The app uses Eagle Router to intelligently select between GPT-4o and GPT-4o-mini based on your query
- **Configurable Parameters**: Adjust P (global vs local weight), N (nearest neighbors), and K (ELO sensitivity) in real-time
- **Model Comparison**: Compare responses from both models side-by-side
- **User Feedback**: Provide feedback on which response is better to improve the router
- **ELO Rankings**: View current model rankings based on accumulated feedback

## Setup

1. **Install dependencies**:
   ```bash
   uv pip install -e .
   # or
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   Create a `.env` file in the project root with:
   ```
   MONGODB_URI=your_mongodb_connection_string
   OPENAI_API_KEY=your_openai_api_key
   ```

3. **Run the app**:
   ```bash
   streamlit run streamlit_app.py
   ```

## How to Use

1. **Configure Router Parameters** (Sidebar):
   - **P (0.0-1.0)**: Balance between global and local scores
     - 0.0 = Pure local (only considers similar queries)
     - 1.0 = Pure global (only considers overall performance)
     - 0.3 = Default (30% global, 70% local)
   
   - **N (1-50)**: Number of nearest neighbors to consider
     - Higher values = more context from similar queries
     - Default = 10
   
   - **K (1-32)**: ELO rating sensitivity
     - Higher values = larger rating changes per match
     - Default = 8

2. **Submit a Query**:
   - Type your prompt in the text area
   - Click "Get Response" to get an answer from the best model

3. **Compare Models**:
   - After receiving a response, click "Compare Models"
   - View responses from both models side-by-side

4. **Provide Feedback**:
   - Choose which response is better (or if they're equal)
   - Your feedback updates the router's model rankings

## Understanding the Router

The Eagle Router uses a hybrid approach:
- **Global Scores**: Overall ELO ratings based on all feedback
- **Local Scores**: Performance on similar queries (using embeddings)
- **Combined Score**: P × Global + (1-P) × Local

This allows the router to:
- Learn general model strengths (global)
- Adapt to query-specific performance (local)
- Improve over time with user feedback

## Models Available

- **GPT-4o-mini**: Faster, more cost-effective model
- **GPT-4o**: More capable, higher quality responses

The router learns which model performs better for different types of queries based on your feedback!
