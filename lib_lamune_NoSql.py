from pymongo import MongoClient


HOST = 'cluster0.boafdid.mongodb.net'
USER = 'Lamune'
PASSWORD = 'tkfkdgo0'

DATABASE_NAME = 'myFirstDatabase'
COLLECTION_NAME = 'openweather'


# DB 컨넥션 가져오기
def get_connection ( COLLECTION_NAME , DATABASE_NAME ) :
    collection = None

    MONGO_URI = f"mongodb+srv://{USER}:{PASSWORD}@{HOST}/{DATABASE_NAME}?retryWrites=true&w=majority"

    client = MongoClient(MONGO_URI)
    # db = client.test
    database = client[DATABASE_NAME]
    collection = database[COLLECTION_NAME]

    return collection

"""
데이터 넣기
collection.insert_one( openweather )
collection.insert_many( octokit )

조회
collection.find()
names = set(collection.distinct( "name" ))

"""