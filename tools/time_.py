from datetime import datetime

def datestamp():
    dt_now = datetime.now()
    return f"{dt_now.year%100:02}{dt_now.month:02}{dt_now.day:02}"