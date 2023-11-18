from discord_webhook import DiscordWebhook, DiscordEmbed
from streamlit import secrets


def send_msg(title: str, description: str, fields: dict, url, color="f8a603"):
    url = url if url != secrets.discord.WEBHOOK_URL_PASSWORD else secrets.discord.WEBHOOK_URL

    webhook = DiscordWebhook(url=url)

    # create embed object for webhook, you can set the color as a decimal (color=242424) or hex (color="03b2f8")
    embed = DiscordEmbed(title=title, description=description, color=color)
    embed.set_author(name="st.dashboard")

    for key, item in fields.items():
        embed.add_embed_field(name=key, value=item)

    # add embed object to webhook
    webhook.add_embed(embed)
    webhook.execute()
