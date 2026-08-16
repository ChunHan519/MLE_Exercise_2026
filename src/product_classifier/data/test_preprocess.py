import pandas as pd

from product_classifier.data.preprocess import preprocess_data
from product_classifier.env import EXPECTED_CATEGORIES

EXPECTED_CATEGORY_LIST = list(EXPECTED_CATEGORIES)


def test_normal_product_with_category():
    df = pd.DataFrame(
        {
            0: ["cookies cakes Chocolate Sandwich Cookies"],
            1: ["Dry Goods & Pantry Staples"],
        }
    )

    result = preprocess_data(df)

    assert result.iloc[0]["product_name"] == (
        "cookies cakes Chocolate Sandwich Cookies"
    )
    assert result.iloc[0]["category"] == "Dry Goods & Pantry Staples"


def test_leading_comma_with_category():
    df = pd.DataFrame(
        {
            0: [""],
            1: ['"Band-Aid Medium Hurt-Free Wrap 2"'],
            2: ["Household & Personal Care"],
        }
    )

    result = preprocess_data(df)

    assert result.iloc[0]["product_name"] == 'Band-Aid Medium Hurt-Free Wrap 2"'
    assert result.iloc[0]["category"] == "Household & Personal Care"


def test_product_name_with_comma():
    df = pd.DataFrame(
        {
            0: ["frozen meals Three Cheese Ziti,Marinara with Meatballs"],
            1: ["Fresh & Perishable Items"],
        }
    )

    result = preprocess_data(df)

    assert result.iloc[0]["product_name"] == (
        "frozen meals Three Cheese Ziti,Marinara with Meatballs"
    )
    assert result.iloc[0]["category"] == "Fresh & Perishable Items"


def test_product_without_category():
    df = pd.DataFrame(
        {
            0: ["Cheesecake, Chocolate Truffle"],
            1: [""],
        }
    )

    result = preprocess_data(df)

    assert result.iloc[0]["product_name"] == (
        "Cheesecake, Chocolate Truffle"
    )
    assert result.iloc[0]["category"] == ""


def test_product_with_multiple_commas_without_category():
    df = pd.DataFrame(
        {
            0: ["kitchen supplies Cake Pans & Lids, Square, 8 Inch"],
            1: [""],
        }
    )

    result = preprocess_data(df)

    assert result.iloc[0]["product_name"] == (
        "kitchen supplies Cake Pans & Lids, Square, 8 Inch"
    )
    assert result.iloc[0]["category"] == ""


def test_strip_whitespace():
    df = pd.DataFrame(
        {
            0: ["  Robust Golden Unsweetened Oolong Tea  "],
            1: ["  Beverages  "],
        }
    )

    result = preprocess_data(df)

    assert result.iloc[0]["product_name"] == (
        "Robust Golden Unsweetened Oolong Tea"
    )
    assert result.iloc[0]["category"] == "Beverages"


def test_clean_quotes():
    df = pd.DataFrame(
        {
            0: ['"Vodka,Triple Distilled,Twist of Vanilla"'],
            1: [""],
        }
    )

    result = preprocess_data(df)

    assert result.iloc[0]["product_name"] == (
        "Vodka,Triple Distilled,Twist of Vanilla"
    )
    assert result.iloc[0]["category"] == ""


def test_remove_duplicates():
    df = pd.DataFrame(
        {
            0: [
                "Chocolate Cookies",
                "Chocolate Cookies",
            ],
            1: [
                "Dry Goods & Pantry Staples",
                "Dry Goods & Pantry Staples",
            ],
        }
    )

    result = preprocess_data(df)

    assert len(result) == 1


def test_expected_category_validation():
    df = pd.DataFrame(
        {
            0: ["Paper Towels 12 Count Mega Rolls"],
            1: ["Household & Personal Care"],
        }
    )

    result = preprocess_data(df)

    assert result.iloc[0]["product_name"] == "Paper Towels 12 Count Mega Rolls"
    assert result.iloc[0]["category"] in EXPECTED_CATEGORY_LIST
    assert result.iloc[0]["category"] == "Household & Personal Care"