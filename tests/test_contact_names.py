import sqlite3
from database_access import open_database
from contact_names import contact_names


def test_local_remarks_nickname_conflicts_and_no_body_inference(tmp_path):
    path=tmp_path/'contacts.db'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE contacts(qq TEXT,remark TEXT,nickname TEXT)')
        db.executemany('INSERT INTO contacts VALUES (?,?,?)',[('1','Synthetic Remark','Nick'),('2','','Synthetic Nick'),('3','Conflict A',''),('3','Conflict B',''),('4','','')])
        db.execute('CREATE TABLE messages(qq TEXT,content TEXT)')
        db.execute("INSERT INTO messages VALUES ('4','My name is Not A Verified Name')")
    with open_database(path) as (db,encrypted,_):
        found=contact_names(db,encrypted,{'1','2','3','4'},local_file=tmp_path/'absent.json')
    assert found=={'1':'Synthetic Remark','2':'Synthetic Nick','3':''}


def test_oversize_names_are_filtered_inside_sql(tmp_path):
    path=tmp_path/'oversize.db'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE contacts(qq TEXT,remark TEXT,nickname TEXT)')
        db.executemany('INSERT INTO contacts VALUES (?,?,?)',[('1','x'*2000000,'Synthetic Nick'),('2','x'*2000000,'y'*2000000)])
    with open_database(path) as (db,encrypted,_):
        found=contact_names(db,encrypted,{'1','2'},local_file=tmp_path/'absent.json')
    assert found=={'1':'Synthetic Nick'}
