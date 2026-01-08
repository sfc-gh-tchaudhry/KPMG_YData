"""
    Example for the bootstrap synthesizer.
"""
from datetime import datetime

from ydata.metadata import Metadata
from ydata.metadata.builder import MetadataConfigurationBuilder
from ydata.synthesizers import FakerSynthesizer
import os

os.environ['YDATA_LICENSE_KEY'] = '74ff0c2a-ae55-41ba-bb00-976bee030b68'
builder = MetadataConfigurationBuilder()
builder.add_column(
    "CustormerID", "numerical", "int", "id"
)
builder.add_column(
    "Name", "string", "string", "name"
)
builder.add_column(
    "Email", "string", "string", "email"
)
builder.add_column(
    "PhoneNumber", "numerical", "int", "phone"
)
builder.add_column(
    "Address", "string", "string", "address"
)
builder.add_column(
    "State", "string", "string", regex="[A-Z]{2}"
)
builder.add_column(
    "PostalCode", "numerical", "int", "zipcode"
)
builder.add_column(
    "Country", "categorical", "string",
    unique=True, categories={"USA": 100}
)
builder.add_column(
    "DateOfBirth", "date", "date",
    min="1950-1-1", max="1990-12-31", format="%Y-%m-%d"
)
builder.add_column(
    "Gender", "categorical", "string",
    categories={"M": 50, "F": 50},
)
builder.add_column(
    "AccountCreateDate", "date", "date",
    min=datetime(2000, 1, 1), max=datetime(2023, 12, 31)
)
builder.add_column(
    "LastPurchaseDate", "date", "date",
    min=datetime(2001, 1, 1), max=datetime(2023, 12, 31)
)
builder.add_column(
    "ProductCategory", "categorical", "string",
    categories={
        "Toys": 50,
        "Clothing": 20,
        "Groceries": 10,
        "Home Goods": 10,
        "Electronics": 10
    },
)
builder.add_column(
    "ProductID", "string", "string",
    regex="[0-9a-zA-Z]{6}-[0-9a-zA-Z]{6}-[0-9a-zA-Z]{6}-[0-9a-zA-Z]{6}"
)
builder.add_column(
    "PurchaseAmount", "numerical", "int",
    min=3000, max=90_000
)
builder.add_column(
    "PurchaseDate", "date", "date",
    min=datetime(2015, 1, 1), max=datetime(2023, 12, 31)
)

meta = Metadata(configuration_builder=builder)
synth = FakerSynthesizer(locale="en")
synth.fit(meta)
sample = synth.sample(100)
print(sample.head(5).T)

# Or it can be created from a dictionary
config = {
    "CustormerID": {
        "datatype": "numerical",
        "vartype": "int",
        "characteristic": "id",
    },
    "Name": {
        "datatype": "string",
        "vartype": "string",
        "characteristic": "name",
    },
    "Email": {
        "datatype": "string",
        "vartype": "string",
        "characteristic": "email",
    },
    "PhoneNumber": {
        "datatype": "string",
        "vartype": "string",
        "characteristic": "phone",
    },
    "Address": {
        "datatype": "string",
        "vartype": "string",
        "characteristic": "address",
    },
    "State": {
        "datatype": "string",
        "vartype": "string",
        "regex": "[A-Z]{2}",
    },
    "PostalCode": {
        "datatype": "numerical",
        "vartype": "int",
        "characteristic": "zipcode",
    },
    "Country": {
        "datatype": "categorical",
        "vartype": "string",
        "categories": {
            "USA": 100,
        },
    },
    "DateOfBirth": {
        "datatype": "date",
        "vartype": "date",
        "min": "1950-1-1",
        "max": "1990-12-31",
        "format": "%Y-%m-%d"
    },
    "Gender": {
        "datatype": "categorical",
        "vartype": "string",
        "categories": {
            "M": 50,
            "F": 50
        },
        "unique": False,
    },
    "AccountCreateDate": {
        "datatype": "date",
        "vartype": "date",
        "min": datetime(2000, 1, 1),
        "max": datetime(2023, 12, 31),
    },
    "LastPurchaseDate": {
        "datatype": "date",
        "vartype": "date",
        "min": datetime(2001, 1, 1),
        "max": datetime(2023, 12, 31),
    },
    "ProductCategory": {
        "datatype": "categorical",
        "vartype": "string",
        "categories": {
            "Toys": 50,
            "Clothing": 20,
            "Groceries": 10,
            "Home Goods": 10,
            "Electronics": 10
        },
        "unique": False,
    },
    "ProductID": {
        "datatype": "string",
        "vartype": "string",
        "regex": "[0-9a-zA-Z]{6}-[0-9a-zA-Z]{6}-[0-9a-zA-Z]{6}-[0-9a-zA-Z]{6}",
    },
    "PurchaseAmount": {
        "datatype": "numerical",
        "vartype": "int",
        "min": 3_000,
        "max": 90_000,
    },
    "PurchaseDate": {
        "datatype": "date",
        "vartype": "date",
        "min": datetime(2015, 1, 1),
        "max": datetime(2023, 12, 31)
    },
}

builder = MetadataConfigurationBuilder(config)
meta = Metadata(configuration_builder=builder)
print(meta)
synth = FakerSynthesizer(locale="en")
synth.fit(meta)
sample = synth.sample(100)

print(sample)
print(sample.to_pandas().isna().sum())

sample.to_pandas().to_csv("y_synthetic_data.csv", index=False)