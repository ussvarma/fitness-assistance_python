import psycopg2

# Connect to your PostgreSQL database on a remote server
conn = psycopg2.connect(host="127.0.0.1", port="5432", dbname="user_details", user="postgres", password="p@ssw0rd")

# Open a cursor to perform database operations
cur = conn.cursor()

# Execute a test query
# cur.execute("CREATE TABLE details(id SERIAL primary key,name VARCHAR(25),username varchar(25),height VARCHAR(25),password VARCHAR(25))")
# conn.commit()


cur.execute("CREATE TABLE user_logindetails(id SERIAL primary key,username varchar(25),pushup int,biceps int,squats int,timeofexercise timestamptz,login timestamptz, logout timestamptz)")
conn.commit()


# name=input("enter name: ")
# password = input("enter password: ")
# with conn:
#
#     cur.execute(f"SELECT * FROM details WHERE name=%(name)s AND password=%(password)s", {'name': name,'password':password})
#     rows = cur.fetchall()
#     print(rows)
#     for row in rows:
#         print(f"{row[0]} {row[1]} {row[2]}")

