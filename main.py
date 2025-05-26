import os
import json
import traceback
from dotenv import load_dotenv
from functions import query_knowledgebase, tool_definitions
from system_prompt import get_system_prompt, ANSWER_PROMPT
import openai

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_bot():
    print("🤖 שוחח עם הבוט. הקלד 'exit' כדי לצאת.")

    while True:
        user_input = input("👤 אתה: ").strip()
        if user_input.lower() == "exit":
            break

        try:
            # Step 1: Ask LLM what tool to call
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": get_system_prompt(tool_definitions)},
                    {"role": "user", "content": user_input}
                ],
                tools=tool_definitions,
                tool_choice="auto"
            )

            tool_call = response.choices[0].message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            if tool_name == "query_knowledgebase":
                tool_result = query_knowledgebase(**tool_args)

                # Step 2: Stream the follow-up response
                follow_up = client.chat.completions.create(
                    model="gpt-4o",
                    stream=True,
                    messages=[
                        {"role": "system", "content": ANSWER_PROMPT},
                        {"role": "user", "content": user_input},
                        response.choices[0].message,
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": json.dumps(tool_result, ensure_ascii=False)
                        }
                    ]
                )

                print("🤖 בוט: ", end="", flush=True)
                for chunk in follow_up:
                    if chunk.choices[0].delta.content:
                        print(chunk.choices[0].delta.content, end="", flush=True)
                print()

        except Exception as e:
            print("❌ שגיאה:")
            print(traceback.format_exc())


if __name__ == "__main__":
    run_bot()
