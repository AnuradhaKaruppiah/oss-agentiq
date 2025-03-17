
1. Create a new clean space for testing:
```bash
rm -rf testing
mkdir testing
cd testing
```

2. Install pypi packages:
```bash
python -m venv env
source env/bin/activate
pip install agentiq[langchain]
```

3. Clone repo:
```bash
git clone git@github.com:NVIDIA/AgentIQ.git agentiq
cd agentiq
```

4. Install dependencies for profiling:
```bash
pip install agentiq[profiling]
```

5. Install example:
```bash
pip install ./examples/simple
```

6. (Optional) To verify agentiq is from pip:
```bash
pip show agentiq
```

6. Run example:
```bash
aiq run --config_file examples/simple/configs/config.yml --input "What is LangSmith?"
```

7. Evaluate example:
```bash
aiq eval --config_file=examples/simple/configs/eval_config.yml
```

8. Test example with serve:
Start the server:
```bash
aiq serve --config_file examples/simple/configs/config.yml
```

Test inputs:
```bash
curl --request POST \
  --url http://localhost:8000/generate \
  --header 'Content-Type: application/json' \
  --data '{
    "input_message": "What is langsmith?",
    "use_knowledge_base": true
}
```
