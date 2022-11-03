import os
import sqlite3
from src import Part_1



QUERY = "INSERT INTO Review VALUES( ? , ? , ? , ? );"
try :
    # conn 생성
    DATABASE_PATH = os.path.join(os.getcwd(), 'scrape_data.db')
    conn = sqlite3.connect(DATABASE_PATH)
    cur = conn.cursor()

    for idx , review in enumerate(reviews) :
        value = ( idx , review['review_text'],  review['review_star'] , movie_title  )
        
        cur.execute(QUERY,value )

    conn.commit()
        
except Exception as e:
    print ( f"에러 발생 {e}")
finally:
    conn.commit()
    conn.close()        