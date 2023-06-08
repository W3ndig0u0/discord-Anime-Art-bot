import discord
import response


async def send_message(message, user_message, is_private):
    try:
        bot_response = response.handle_response(user_message)
        if is_private:
            await message.author.send(bot_response)
        else:
            await message.channel.send(bot_response)
    except Exception as e:
        print(e)


def run_discord():
    TOKEN = "MTExNjQ4NTQwNDEyODk2ODgxNw.G6vm03.Z-GMCFEoDAlyDa-MqTT1bond4cHFRV_sqDONCU"
    intents = discord.Intents.default()
    intents.message_content = True

    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(str(client.user) + " is now running")

    @client.event
    async def on_message(message):
        if message.author == client.user:
            return

        username = str(message.author)
        user_message = str(message.content)
        channel = str(message.channel)

        print(username + " said " + user_message + " in " + channel)

        if user_message and user_message[0] == "!":
            user_message = user_message[1:]
            await send_message(message, user_message, is_private=True)
        else:
            await send_message(message, user_message, is_private=False)

    client.run(TOKEN)


run_discord()
