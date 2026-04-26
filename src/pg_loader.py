from sqlalchemy import create_engine

from data import res_data

engine = create_engine("postgresql://tuser:12345@localhost:5432/mydb")

res_df = res_data.copy()

res_df.to_sql("iris_data", engine, if_exists="replace", index=False)
