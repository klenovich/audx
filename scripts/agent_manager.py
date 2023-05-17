from llm_utils import create_chat_completion

next_key = 0
agents = {}  # key, (task, full_message_history, model)


def interact_with_agent(model, messages):
    # Interact with the model while enforcing the token limit
    agent_reply = create_chat_completion(
        model=model,
        messages=messages
    )
    return agent_reply


def create_agent(task, prompt, model):
    global next_key
    global agents

    messages = [{"role": "user", "content": prompt}, ]
    agent_reply = interact_with_agent(model, messages)
    messages.append({"role": "assistant", "content": agent_reply})

    key = next_key
    next_key += 1

    agents[key] = (task, messages, model)

    return key, agent_reply


def message_agent(key, message):
    global agents

    task, messages, model = agents[int(key)]
    messages.append({"role": "user", "content": message})
    agent_reply = interact_with_agent(model, messages)
    messages.append({"role": "assistant", "content": agent_reply})

    return agent_reply


def list_agents():
    global agents
    return [(key, task) for key, (task, _, _) in agents.items()]


def delete_agent(key):
    global agents

    try:
        del agents[int(key)]
        return True
    except KeyError:
        return False