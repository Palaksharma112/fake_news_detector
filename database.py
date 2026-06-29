import sqlite3

def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, username TEXT, password TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY, username TEXT, news TEXT, result TEXT)''')

    conn.commit()
    conn.close()


def add_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO users (username, password) VALUES (?,?)",
              (username, password))
    conn.commit()
    conn.close()


def get_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (username, password))
    return c.fetchone()


def add_history(username, news, result):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    c.execute(
        "INSERT INTO history (username, news, result) VALUES (?, ?, ?)",
        (username, news, result)
    )

    conn.commit()
    conn.close()

    

def get_history(username):

    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    c.execute("""
        SELECT news,
               result
        FROM history
        WHERE username=?
        ORDER BY id DESC
    """, (username,))

    data = c.fetchall()
    conn.close()
    return data



def get_stats(username):

    conn = sqlite3.connect("users.db")
    cur = conn.cursor()

    total_checks = cur.execute(
        "SELECT COUNT(*) FROM history WHERE username=?",
        (username,)
    ).fetchone()[0]

    real_count = cur.execute(
        "SELECT COUNT(*) FROM history WHERE username=? AND result LIKE '%REAL%'",
        (username,)
    ).fetchone()[0]

    fake_count = cur.execute(
        "SELECT COUNT(*) FROM history WHERE username=? AND result LIKE '%FAKE%'",
        (username,)
    ).fetchone()[0]

    uncertain_count = cur.execute(
        "SELECT COUNT(*) FROM history WHERE username=? AND result LIKE '%UNCERTAIN%'",
        (username,)
    ).fetchone()[0]

    conn.close()

    return (
        total_checks,
        real_count,
        fake_count,
        uncertain_count
    )