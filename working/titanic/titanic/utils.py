def try_except(error_stop=True):
    def inner(func):
    
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"ERROR: {e}")
                if error_stop: raise SystemExit
                else: return None
        return wrapper
    
    return inner