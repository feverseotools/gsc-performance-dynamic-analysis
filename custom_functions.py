import pandas as pd
import numpy as np

def top_queries_by_clicks(df, n=10):
    """
    Retorna las 'n' queries con mayor número de clics.
    """
    # Asumiendo que el DataFrame tiene una columna "query" y "clicks"
    return df.groupby("query", as_index=False)["clicks"].sum().sort_values("clicks", ascending=False).head(n)

def avg_ctr_by_page(df):
    """
    Retorna el CTR promedio por página.
    """
    # Asumiendo que el DataFrame tiene columnas "page" y "ctr"
    return df.groupby("page", as_index=False)["ctr"].mean().sort_values("ctr", ascending=False)

def filter_by_country(df, country_code):
    """
    Filtra el DataFrame por un código de país (por ejemplo, 'ES').
    """
    # Asumiendo que el DataFrame tiene una columna "country"
    return df[df["country"] == country_code].copy()

def summarize_data(df):
    """
    Retorna un resumen estadístico genérico del DataFrame.
    """
    return df.describe(include='all')
