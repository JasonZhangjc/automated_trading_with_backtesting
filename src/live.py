# Collect trading signals and send to emails



import yfinance as yf
from apscheduler.schedulers.blocking import BlockingScheduler
from email.message import EmailMessage
import ssl
import smtplib
from credentials import gmail_user, gmail_password



def get_data(symbol="AAPL"):
    # Define the ticker symbol
    ticker_symbol = symbol

    # Create a ticker object
    ticker = yf.Ticker(ticker_symbol)

    # Download historical data
    return ticker.history(period="2d", interval='1d')



def test_engulfing(df):
    last_open = df.iloc[-1, :].Open
    last_close = df.iloc[-1, :].Close
    previous_open = df.iloc[-2, :].Open
    previous_close = df.iloc[-2, :].Close

    if (previous_open < previous_close
        and last_open > previous_close
        and last_close < previous_open):
        return 1  # Bearish Engulfing Pattern

    elif (previous_open > previous_close
          and last_open < previous_close
          and last_close > previous_open):
        return 2  # Bullish Engulfing Pattern
    else:
        return 0  # No Engulfing Pattern



# Send live signal
def some_job_single():
    msg="Trading Signal Message \n"
    historical_data = get_data()

    if test_engulfing(historical_data)==1:
        msg = str("the signal is 1 bearish")

    elif test_engulfing(historical_data)==2:
        msg = str("the signal is 2 bullish")

    em['From'] = gmail_user
    em['To'] = gmail_user
    em['Subject'] = subject
    em.set_content(msg)

    context = ssl.create_default_context()

    server = smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context)
    server.ehlo()
    server.login(gmail_user, gmail_password)
    server.sendmail(gmail_user, gmail_user, em.as_string())
    server.close()



# Send live signal
def some_job_multiple(symbols):
    msg="Trading Signal Message \n"
    historical_data = get_data()

    if test_engulfing(historical_data)==1:
        msg = str("the signal is 1 bearish")

    elif test_engulfing(historical_data)==2:
        msg = str("the signal is 2 bullish")

    em['From'] = gmail_user
    em['To'] = gmail_user
    em['Subject'] = subject
    em.set_content(msg)

    context = ssl.create_default_context()

    server = smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context)
    server.ehlo()
    server.login(gmail_user, gmail_password)
    server.sendmail(gmail_user, gmail_user, em.as_string())
    server.close()



if __name__ == '__main__':
    # df = get_data()
    # test_engulfing(df)

    # single stock
    em = EmailMessage()
    gmail_user = gmail_user
    gmail_password = gmail_password
    subject = 'info signal'
    some_job_single()

    # multiple stock
    symbols =  ['AAPL', 'NVDA', 'PYPL']
    some_job_multiple(symbols)

    # schedule the signal message to gmail
    scheduler = BlockingScheduler(job_defaults={'misfire_grace_time': 15*60})
    scheduler.add_job(some_job_multiple, 'cron', day_of_week='mon-fri', hour=0, minute=0, timezone=utc)
    scheduler.start()