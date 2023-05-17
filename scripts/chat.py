import time
import openai
from dotenv import load_dotenv
from config import Config
import token_counter

cfg = Config()

from llm_utils import create_chat_completion


def create_chat_message(role, content):
    return {"role": role, "content": content}


def generate_context(prompt, relevant_memory, full_message_history, model):
    current_context = [
        create_chat_message(
            "system", prompt), create_chat_message(
            "system", f"Permanent memory: {relevant_memory}")]

    next_message_to_add_index = len(full_message_history) - 1
    insertion_index = len(current_context)
    current_tokens_used = token_counter.count_message_tokens(current_context, model)
    return next_message_to_add_index, current_tokens_used, insertion_index, current_context


def chat_with_ai(
        prompt,
        user_input,
        full_message_history,
        permanent_memory,
        token_limit,
        debug=False,
        model=None):
    if model is None:
        model = cfg.fast_llm_model

    while True:
        try:
            if debug:
                print(f"Token limit: {token_limit}")
            send_token_limit = token_limit - 1000

            relevant_memory = permanent_memory.get_relevant(str(full_message_history[-5:]), 10)

            if debug:
                print('Memory Stats: ', permanent_memory.get_stats())

            next_message_to_add_index, current_tokens_used, insertion_index, current_context = generate_context(
                prompt, relevant_memory, full_message_history, model)

            while current_tokens_used > 2500:
                relevant_memory = relevant_memory[1:]
                next_message_to_add_index, current_tokens_used, insertion_index, current_context = generate_context(
                    prompt, relevant_memory, full_message_history, model)

            current_tokens_used += token_counter.count_message_tokens([create_chat_message("user", user_input)], model)

            while next_message_to_add_index >= 0:
                message_to_add = full_message_history[next_message_to_add_index]
                tokens_to_add = token_counter.count_message_tokens([message_to_add], model)
                if current_tokens_used + tokens_to_add > send_token_limit:
                    break

                current_context.insert(insertion_index, full_message_history[next_message_to_add_index])
                current_tokens_used += tokens_to_add
                next_message_to_add_index -= 1

            current_context.extend([create_chat_message("user", user_input)])
            tokens_remaining = token_limit - current_tokens_used

            if debug:
                print(f"Token limit: {token_limit}")
                print(f"Send Token Count: {current_tokens_used}")
                print(f"Tokens remaining for response: {tokens_remaining}")
                print("------------ CONTEXT SENT TO AI ---------------")
                for message in current_context:
                    if message["role"] == "system" and message["content"] == prompt:
                        continue
                    print(
                        f"{message['role'].capitalize()}: {message['content']}")
                    print()
                print("----------- END OF CONTEXT ----------------")

            assistant_reply = create_chat_completion(
                model=model,
                messages=current_context,
                max_tokens=tokens_remaining,
            )

            full_message_history.append(
                create_chat_message(
                    "user", user_input))
            full_message_history.append(
                create_chat_message(
                    "assistant", assistant_reply))

            return assistant_reply
        except openai.error.RateLimitError:
            print("Error: ", "API Rate Limit Reached. Waiting 10 seconds...")
            time.sleep(10)