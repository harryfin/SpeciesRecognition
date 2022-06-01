import pytest
import utils as ut
from datasetCUB.Cub_class.class_cub import Cub2011

root = ut.get_root()
cub_root = ut.get_root_CUB(root)

Datenset = Cub2011(cub_root)
data = Datenset.data


@pytest.mark.parametrize(
    "col, row, value",
    [
        ["img_id", 8, 9],
        [
            "filepath",
            8,
            "001.Black_footed_Albatross/Black_Footed_Albatross_0010_796097.jpg",
        ],
        ["img_class", 8, 1],
        ["img_class", 11783, 200],
        [
            "attribute_ids",
            11783,
            [
                2,
                15,
                30,
                45,
                55,
                64,
                79,
                85,
                86,
                98,
                111,
                127,
                146,
                152,
                164,
                173,
                174,
                188,
                203,
                219,
                236,
                237,
                244,
                245,
                254,
                267,
                281,
                290,
                306,
                309,
            ],
        ],
        ["part_ids", 11783, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15]],
    ],
)
def test_load_data(col, row, value):
    assert data[col][row] == value
