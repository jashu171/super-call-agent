# BotCampus AI Support Agent

LiveKit voice agent for BotCampus AI customer support and booking.

## Deployment to LiveKit Cloud

### Prerequisites
1. LiveKit Cloud account
2. LiveKit CLI installed: `brew install livekit-cli` (macOS) or download from [LiveKit releases](https://github.com/livekit/livekit-cli/releases)

### Step 1: Install LiveKit CLI
```bash
# macOS
brew install livekit-cli

# Or download binary from GitHub releases
```

### Step 2: Authenticate with LiveKit Cloud
```bash
livekit-cli cloud auth
```

### Step 3: Set up secrets in LiveKit Cloud Dashboard
Go to your LiveKit Cloud dashboard and add these secrets:

**Secret: `livekit-config`**
- `url`: Your LiveKit WebSocket URL (e.g., `wss://your-project.livekit.cloud`)
- `api_key`: Your LiveKit API key
- `api_secret`: Your LiveKit API secret

**Secret: `agent-secrets`**
- `google_api_key`: Your Google AI API key
- `deepgram_api_key`: Your Deepgram API key

### Step 4: Deploy the agent
```bash
# Deploy to LiveKit Cloud
livekit-cli cloud deploy agent.yaml

# Or use the deploy command
livekit-cli deploy
```

### Step 5: Monitor your agent
```bash
# Check agent status
livekit-cli cloud agent list

# View agent logs
livekit-cli cloud agent logs botcampus-support-agent
```

## Local Development

### Setup
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run locally
```bash
# Make sure .env file has all required variables
python agent.py
```

## Environment Variables

Required in `.env` for local development:
- `LIVEKIT_URL`: WebSocket URL
- `LIVEKIT_API_KEY`: API key
- `LIVEKIT_API_SECRET`: API secret
- `GOOGLE_API_KEY`: Google AI (Gemini) API key
- `DEEPGRAM_API_KEY`: Deepgram API key
- `N8N_WEBHOOK_URL`: N8N booking webhook
- `N8N_AVAILABILITY_WEBHOOK_URL`: N8N availability webhook

## Features
- Voice-based customer support
- Course information and booking
- Availability checking
- User information collection
- Natural conversation flow
- Deepgram text intelligence (sentiment, topics, intents)
