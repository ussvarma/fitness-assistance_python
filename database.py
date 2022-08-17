import psycopg2

# Connect to your PostgreSQL database on a remote server
conn = psycopg2.connect(host="127.0.0.1", port="5432", dbname="user_details", user="postgres", password="p@ssw0rd")

# Open a cursor to perform database operations
cur = conn.cursor()

# Execute a test query
cur.execute("CREATE TABLE details(id SERIAL primary key,name VARCHAR(25),username varchar(25),height VARCHAR(25),weight VARCHAR(25),password VARCHAR(25))")
cur.execute("CREATE TABLE user_logindetails(id SERIAL primary key,username varchar(25),login timestamptz,pushup float,biceps float,squats float,lunges float,short_head_biceps float)")
#cur.execute("CREATE TABLE biceps(username varchar(25),starttime timestamptz,endtime timestamptz)")
conn.commit()


