import argparse
import logging
from typing import Dict

from telegram import ReplyKeyboardMarkup, Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    CallbackContext,
    CallbackQueryHandler,
)

from Recommendation_engine.inference import predict_cuisine, get_similar_recipes
from recogonition_engine.inference import classify_image

# Create the parser
# my_parser = argparse.ArgumentParser(description='MY App')

# # Add the arguments
# my_parser.add_argument('1958946387:AAG-yOpUEiBSArFUfzel17BRZqBvIviU-ts', metavar='Message', type=str, help='The token given by Fatherbot')


# Enable logging
logging.basicConfig(
    #filename= 'telgramBot.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# Main interactions
CHOOSING, GET_TEXT, GET_IMAGE = range(3)
# Callback data
CALLBACK1, CALLBACK2 = range(3,5)

reply_keyboard = [
    ['Show ingredients', 'Get recipes'],
    ['Remove item', 'Done'],
]
markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=False)

def start(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    logger.info(f"{user.first_name}: Start")

    context.user_data['chat_id'] = update.message.chat_id

    update.message.reply_text(
        "Enter the ingredients",
        reply_markup=markup,
    )
    return CHOOSING

def get_basket_txt(list_ingredients):
    txt = 'current ingredients:\n'
    for ingredient in list_ingredients:
        txt += f'   - {ingredient}\n'
    return txt

def received_image_information(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    photo_file = update.message.photo[-1].get_file()
    photo_file.download('infer_image.png')
    #logger.info("Photo of %s: %s", user.first_name, 'infer_image.jpg')
    update.message.reply_text(
        'Image is processing'
    )

    user_data = context.user_data
    
    # Infer image prediction
    ingredient = classify_image('infer_image.png')
    
    keyboard = [
        [
            InlineKeyboardButton(ingredient[0], callback_data=ingredient[0]),
            InlineKeyboardButton(ingredient[1], callback_data=ingredient[1]),
            InlineKeyboardButton(ingredient[2], callback_data=ingredient[2])],
        [
            InlineKeyboardButton(ingredient[3], callback_data=ingredient[3])
        ]
        #     InlineKeyboardButton(ingredient[4], callback_data=ingredient[4]),
        # ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    # Send message with text and appended InlineKeyboard
    update.message.reply_text("Top Predicted ingredients and select one of them", reply_markup=reply_markup)

    return CALLBACK1

def button1(update: Update, context: CallbackContext) -> int:
    logger.info(f": button1")

    query = update.callback_query
    query.answer()

    user_data = context.user_data    
    if 'ingredients_list' not in user_data:
        user_data['ingredients_list'] = [query.data]
    else:
        user_data['ingredients_list'].append(query.data)

    query.edit_message_text(text=f"Ok you selected: {query.data}")
    
    txt = get_basket_txt(user_data['ingredients_list'])
    context.bot.send_message(chat_id=context.user_data['chat_id'], text=txt)

    return CHOOSING

def recipes_query(update: Update, context: CallbackContext) -> int:
    """ Get recipes """
    user = update.message.from_user
    logger.info(f"{user.first_name}: recipes_query")

    user_data = context.user_data

    input_text = ' '.join(user_data['ingredients_list'])

    # Predict cuisine
    cuisine = predict_cuisine(input_text)

    keyboard = [
        [
            InlineKeyboardButton(cuisine[0], callback_data=cuisine[0]),
            InlineKeyboardButton(cuisine[1], callback_data=cuisine[1]),
            InlineKeyboardButton(cuisine[2], callback_data=cuisine[2])],
        [
            InlineKeyboardButton(cuisine[3], callback_data=cuisine[3]),
            InlineKeyboardButton(cuisine[4], callback_data=cuisine[4]),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    # Send message with text and appended InlineKeyboard
    update.message.reply_text("cuisines listed", reply_markup=reply_markup)

    return CALLBACK2

def button2(update: Update, context: CallbackContext) -> int:
    #user = update.message.from_user
    logger.info(f"button2")

    query = update.callback_query
    query.answer()

    # Get recipes
    recipes = get_similar_recipes(context.user_data['ingredients_list'], query.data)

    sep = '\n\n'
    for index, row in recipes.iterrows():

        title = 'Title: ' + row['title'] 
        ingredients=''
        list_ing = row['ingredients'].replace('ADVERTISEMENT', '').strip('][').split(', ')
        for ingredient in list_ing:
            ingredients+= ingredient.replace("'", "") + '\n'
        ingredients = 'Ingredients: ' + '\n' + ingredients
        instructions = 'Instruction: '+ '\n' + row['instructions']

        txt = title + sep + ingredients + sep + instructions

        context.bot.send_message(chat_id=context.user_data['chat_id'], text=txt)

    return CHOOSING


def show_basket(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    logger.info(f"{user.first_name}: show_basket")

    user_data = context.user_data
    
    txt = get_basket_txt(user_data['ingredients_list'])
    
    update.message.reply_text(
        txt,
        reply_markup=markup,
    )
    return CHOOSING

def received_text_information(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    logger.info(f"{user.first_name}: received_text_information")

    user_data = context.user_data
    text = update.message.text
    
    if 'ingredients_list' not in user_data:
        user_data['ingredients_list'] = [text]
    else:
        user_data['ingredients_list'].append(text)

    txt = get_basket_txt(user_data['ingredients_list'])
    update.message.reply_text(
        txt,
        reply_markup=markup,
    )
    return CHOOSING

def remove_item(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    logger.info(f"{user.first_name}: delete_one__item")

    user_data = context.user_data
    if 'ingredients_list' in user_data:
        del user_data['ingredients_list'][-1]
    
    introduction = 'ingredient removed from the list. '
    txt = get_basket_txt(user_data['ingredients_list'])
    update.message.reply_text(
        introduction + txt,
        reply_markup=markup,
    )
    return CHOOSING

def done(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    logger.info(f"{user.first_name}: done")

    user_data = context.user_data
    if 'ingredients_list' in user_data:
        del user_data['ingredients_list']

    update.message.reply_text(
        f"Session ended"
    )

    user_data.clear()
    return ConversationHandler.END    

def main(bot_token) -> None:
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    updater = Updater(bot_token, use_context=True)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # Add conversation handler with the states CHOOSING, TYPING_CHOICE and TYPING_REPLY
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            CHOOSING: [
                MessageHandler(Filters.photo & ~(Filters.command | Filters.regex('^(Done|Get recipes|Show ingredients|Remove item)$')), received_image_information),
                MessageHandler(Filters.text & ~(Filters.command | Filters.regex('^(Done|Get recipes|Show ingredients|Remove item)$')), received_text_information),
                MessageHandler(Filters.regex('^Get recipes$'), recipes_query),
                MessageHandler(Filters.regex('^Show ingredients$'), show_basket),
                MessageHandler(Filters.regex('^Remove item$'), remove_item),
            ],
            CALLBACK1: [
                CallbackQueryHandler(button1)],
            CALLBACK2: [
                CallbackQueryHandler(button2)],
        },
        fallbacks=[MessageHandler(Filters.regex('^Done$'), done)],
        per_message=False,
    )

    dispatcher.add_handler(conv_handler)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()

if __name__ == '__main__':
    # Execute the parse_args() method
    #args = my_parser.parse_args()
    token = "1958946387:AAG-yOpUEiBSArFUfzel17BRZqBvIviU-ts"
    main(token)