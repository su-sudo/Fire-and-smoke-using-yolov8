from twilio.rest import Client
def generate(phone):
    # set Twilio account SID, auth token, and phone number
    account_sid = 'ACc595b2f81283079783ea3bff0c8b3b33'
    auth_token = '337ecbc6afca5614eab72a7f81f50fd3'
    twilio_number = '+13184966152'

    # create Twilio client
    client = Client(account_sid, auth_token)

    # send SMS message containing OTP
    if True:
        message = client.messages.create(
            body=f'Fire detected!!!',
            from_=twilio_number,
            to="+91"+phone
        )

generate("9539983198")