import discord
import response
from PIL import Image
import io
from art import AnimeArtist


async def send_message(message, user_message, is_private):
    try:
        bot_response = response.handle_response(user_message)
        if is_private:
            await message.author.send(bot_response)
        else:
            await message.channel.send(bot_response)
    except Exception as e:
        error_message = "An error occurred: " + str(e)
        if is_private:
            await message.author.send(error_message)
        else:
            await message.channel.send(error_message)


async def send_image_to_user(message, image):
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        await message.channel.send(file=discord.File(img_byte_arr, "art.png"))
    except Exception as e:
        error_message = "An error occurred while sending the image: " + str(e)
        await message.channel.send(error_message)


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
            if user_message and (
                user_message[0] == "?"
                or (len(user_message) > 1 and user_message[1] == "?")
            ):
                print("GENERATE")
                # Handle input prompt
                input_prompt = (
                    user_message[1:] if user_message[0] == "?" else user_message[2:]
                )

                await send_message(
                    message, "Input prompt set: " + input_prompt, is_private=False
                )
                # Call the generate_art function from AnimeArtist class here
                anime_artist = AnimeArtist()
                height = 512
                width = 512
                num_inference_steps = 100
                eta = 0.1
                negative_prompt = "sketch, duplicate, ugly, huge eyes, text, logo, monochrome, worst face form, worst qualirt, bad autonomy "
                guidance_scale = 6
                seed = -1
                batch_size = 1
                art_model_id = "JingAnimeV2"
                vae_name = "stabilityai/sd-vae-ft-mse"
                save_folder = "./img"
                initial_generation = True

                generating_message = "Generating image..."
                await send_message(message, generating_message, is_private=False)

                save_folder, final_file_number = anime_artist.generate_art(
                    input_prompt,
                    height,
                    width,
                    num_inference_steps,
                    eta,
                    negative_prompt,
                    guidance_scale,
                    save_folder,
                    seed,
                    batch_size,
                    art_model_id,
                    vae_name,
                    initial_generation,
                )
                image_path = f"{save_folder}/{final_file_number}.png"
                image = Image.open(image_path)
                await send_image_to_user(message, image)
            else:
                await send_message(message, user_message, is_private=False)

    client.run(TOKEN)


run_discord()
