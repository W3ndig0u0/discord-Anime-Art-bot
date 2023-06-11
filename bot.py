import response
from PIL import Image
import io
from art import AnimeArtist
import discord


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
                negative_prompt = "Worst quality, bad quality, sketch, duplicate, ugly, huge eyes, text, logo, monochrome, worst face, (bad and mutated hands:1.4), (worst quality:2.0), (low quality:2.0), (blurry:2.0), horror, geometry, bad_prompt, (bad hands), (missing fingers), multiple limbs, bad anatomy, (interlocked fingers:1.3), Ugly Fingers, (extra digit and hands and fingers and legs and arms:1.4), ((2girl)), (deformed fingers:1.3), (long fingers:1.2),(bad-artist-anime), bad-artist, bad hand, extra legs, (worst quality, low quality:2), NSFW,monochrome, zombie,overexposure, watermark,text,bad anatomy,bad hand,((extra hands)),extra fingers,too many fingers,fused fingers,bad arm,distorted arm,extra arms,fused arms,extra legs,missing leg,disembodied leg,extra nipples, detached arm, liquid hand,inverted hand,disembodied limb, oversized head,extra body,extra navel,easynegative,(hair between eyes),sketch, duplicate, ugly, huge eyes, text, logo, worst face, (bad and mutated hands:1.3), (blurry:2.0), horror, geometry, bad_prompt, (bad hands), (missing fingers), multiple limbs, bad anatomy, (interlocked fingers:1.2), Ugly Fingers, (extra digit and hands and fingers and legs and arms:1.4), (deformed fingers:1.2), (too many fingers:1.3),(bad-artist-anime), bad-artist, bad hand, extra legs ,(ng_deepnegative_v1_75t),((hands on head)) ((deformed legs:1.3)) (black background:1.3) (empty background:1.2) (mutaded or blub fingers:1.4), (bad autonomy:1.2) (Worst Quality, Low Quality:1.4), (Poorly Made Bad 3D, Lousy Bad Realistic:1.1), bad mouth form, ((too many fingers:1.1)), too few fingers, bad legs autonomy, (bad hand structure:1.1), ((weird fingers:1.2)), bad thumb, (long fingers:1.3), (glasses:1.2) ((Bad eye form)) (holding anything:1.2), nsfw, lewd, sex, hot, sexy, hentai, nude, ugly smile, bad eyes form, tiny eyes, huge eyes, diffrrent eyes, diffrent eyeslenses, text, watermark, mutated hands, small fingers, long fingers, to many fingers, to few fingers, ugly feets, ugly eyes, tiny eyes, to large eyes, bad eyes form, bad eyes structure"
                guidance_scale = 9
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
