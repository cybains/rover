# services/ingest/connectors/coverage_report_core.py
import os
from collections import defaultdict
from dotenv import load_dotenv
from pymongo import MongoClient

INDICATORS = [
    "NY.GDP.PCAP.KD","NY.GDP.MKTP.KD.ZG","NE.EXP.GNFS.ZS","FP.CPI.TOTL.ZG",
    "SL.UEM.TOTL.ZS","SL.TLF.CACT.ZS","SL.EMP.TOTL.SP.ZS",
    "SP.POP.TOTL","SP.DYN.LE00.IN","SH.XPD.CHEX.PC.CD","SH.IMM.MEAS.ZS",
    "SE.SEC.ENRR","SE.TER.ENRR","SE.ADT.LITR.ZS",
    "IT.NET.USER.ZS","IT.CEL.SETS.P2","IT.NET.BBND.P2","BX.GSR.CCIS.CD",
    "EN.ATM.CO2E.PC","EN.ATM.PM25.MC.M3","EG.ELC.ACCS.ZS","EG.ELC.RNEW.ZS","EG.USE.PCAP.KG.OE",
    "IC.LGL.CRED.XQ","IC.BUS.NDNS.ZS","FS.AST.PRVT.GD.ZS","TX.VAL.TECH.MF.ZS","IP.JRN.ARTC.SC",
    "PA.NUS.PPPC.RF","SH.STA.HOMIC.ZS","SP.URB.TOTL.IN.ZS",
]

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI: raise SystemExit("MISSING MONGO_URI")
client = MongoClient(MONGO_URI)
col = client["worldbank_core"]["observations"]

def to_int(s):
    try: return int(s)
    except: return None

print("code | latest_year | countries_at_latest | countries_5y | avg_pts_per_country")
print("-----|-------------|---------------------|--------------|---------------------")

for code in INDICATORS:
    # latest year with non-null value
    pipe_latest = [
        {"$match": {"indicator.id": code, "value": {"$ne": None}}},
        {"$addFields": {"date_i": {"$toInt": "$date"}}},
        {"$group": {"_id": None, "latest": {"$max": "$date_i"}}}
    ]
    latest_doc = list(col.aggregate(pipe_latest))
    if not latest_doc:
        print(f"{code} | n/a | 0 | 0 | 0.0")
        continue
    latest = latest_doc[0]["latest"]

    # countries at latest year
    pipe_latest_cov = [
        {"$match": {"indicator.id": code, "value": {"$ne": None}, "date": str(latest)}},
        {"$group": {"_id": "$country.id"}}
    ]
    countries_latest = len(list(col.aggregate(pipe_latest_cov)))

    # countries with any value in last 5y window
    pipe_5y = [
        {"$match": {"indicator.id": code, "value": {"$ne": None}, "date": {"$gte": str(latest-4), "$lte": str(latest)}}},
        {"$group": {"_id": "$country.id"}}
    ]
    countries_5y = len(list(col.aggregate(pipe_5y)))

    # avg points per country overall
    pipe_avg = [
        {"$match": {"indicator.id": code, "value": {"$ne": None}}},
        {"$group": {"_id": "$country.id", "n": {"$sum": 1}}},
        {"$group": {"_id": None, "avg": {"$avg": "$n"}}}
    ]
    avg_doc = list(col.aggregate(pipe_avg))
    avg_pts = round(avg_doc[0]["avg"], 1) if avg_doc else 0.0

    print(f"{code} | {latest} | {countries_latest} | {countries_5y} | {avg_pts}")
