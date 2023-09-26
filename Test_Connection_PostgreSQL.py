
import psycopg2

host = "127.0.0.1"
dbname = "parking"
user = "postgres"
password = "9999"
sslmode = "disable"

con_string = "host={0} user={1} dbname={2} password={3} sslmode={4}".format(host, user, dbname, password, sslmode)

cnxn = psycopg2.connect(con_string)
crsr = cnxn.cursor()
sql = "SELECT NOW()"
crsr.execute(sql)
row = crsr.fetchone()
tanggal = row[0]
print (tanggal)
cnxn.commit()
crsr.close()
cnxn.close()
		

		
		
