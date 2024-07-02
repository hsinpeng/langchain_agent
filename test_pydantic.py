import sys
from datetime import datetime
from pydantic import BaseModel, PositiveInt
from pydantic import ValidationError

class User(BaseModel):
    id: int  
    name: str = 'John Doe'  
    signup_ts: datetime | None  
    tastes: dict[str, PositiveInt]  

def main():
    try:
        print("Hello, World!")
    
        run_option = 0
        match run_option:
            case 0:
                external_data = {
                    'id': 123,
                    #'signup_ts': '2019-06-01 12:22',  
                    'signup_ts': None,  
                    'tastes': {
                        'wine': 9,
                        'cheese': 7,  
                        'cabbage': '1',  
                    },
                }

                user = User(**external_data)  

                print(user.id) 
                print(user.model_dump())  
            case 1:
                external_data = {'id': 'not an int', 'tastes': {}}
                User(**external_data)
            case _:
                print(f'Error: Wrong run_option({run_option})!')

    except ValueError as ve:
        print(ve.errors())
        return str(ve)

if __name__ == "__main__":
    sys.exit(main())