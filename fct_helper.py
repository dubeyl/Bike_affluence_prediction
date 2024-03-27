from datetime import datetime

def ms_to_dt(datetime_ms):
    # Convert milliseconds to seconds
    seconds = datetime_ms / 1000.0

    # Convert seconds to datetime
    dt_object = datetime.fromtimestamp(seconds)

    print("Datetime:", dt_object)
    return(dt_object)
    
#ms_to_dt(1709238505598) # example
