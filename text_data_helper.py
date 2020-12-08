"""
Library to assist in loading and analyzing CSV and JSON exports from the PhoneView macOS app and Facebook.
"""

"""
Code adapted from Jérémie Lumbroso, Hien Pham, and Daniel Goodman. Authored by Daniel Wey
"""

# Standard Python library
import collections
import csv
import datetime
import enum
import io
import re
import textwrap
import typing
import emojis
import emoji
import random

# Libraries for JSON File
import json
import os
import shutil
import pandas as pd

# Pandas library: May need to be installed
try:
    import pandas as pd
except ImportError as err:
    err.message = textwrap.dedent(
        """
        You are missing the `pandas` package. Install it before proceeding.
    
        See: https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html
    
        Original error message:
            {original_msg}
        """.format(original_msg=err.message))


# Some hard-coded constants

DEFAULT_PHONE_REGION = "US"

# Some hard-coded constants having to do with the PhoneView file format

PHONEVIEW_INTERNAL_DB_FIELDS = ["id", "date", "address", "text", "flags"]
PHONEVIEW_INTERNAL_RECEIVED = "2"
PHONEVIEW_INTERNAL_SENT = "3"
PHONEVIEW_EXTERNAL_RECEIVED = "Received"

# Constants for iMessage reactions
SPA_REACTIONS = ["exclamó por “", "le dio risa “", "le gustó “", "dudó sobre “", "le encantó “"]
ENG_REACTIONS = ["emphasized “", "laughed at “", "liked “", "questioned “", "loved “"]

# The elementary record type for an entry of PhoneView message log
PhoneViewMsgData = typing.TypedDict(
    "PhoneViewMsgData",
    {
        "timestamp": datetime.datetime,
        "inbound":   bool,
        "length":    int,
        "content":   str,
        "number":    str,
        "name":      str,
        "type":      str,
    },
    total=False
)

# Different states for analysis of the data
class PlotStyle(enum.Enum):
    COUNT = 'count'
    VOLUME = 'volume'
    KEYWORD = 'keyword'
    REACTIONS = 'reaction'


def _load_internal_phoneview_msg_file(
        raw_rows: typing.List[typing.List[str]],
        header: typing.List[str]
) -> typing.List[PhoneViewMsgData]:

    records = []
    
    for row in raw_rows:
        row_dict = dict(zip(header, row))
        
        record = collections.OrderedDict()
        
        if "date" in row_dict:
            # the timestamp is in Unix format
            timestamp = int(row_dict["date"])
            record["timestamp"] = datetime.datetime.fromtimestamp(timestamp)
        
        if "flags" in row_dict:
            record["inbound"] = (row_dict["flags"] == PHONEVIEW_INTERNAL_RECEIVED)
        
        text = row_dict.get("text", "")
        record["length"] = len(text)
        record["content"] = text
        
        if "address" in row_dict:
            record["number"] = row_dict["address"]
        
        if "grouptitle" in row_dict and row_dict.get("grouptitle") != "":
            record["name"] = row_dict["grouptitle"]
        
        records.append(record)
    
    return records


def _load_external_phoneview_msg_file(
        raw_rows: typing.List[typing.List[str]]
) -> typing.List[PhoneViewMsgData]:

    records = []

    for row in raw_rows:
        # Message has this format (last row is missing from old versions):
        # ['Received',
        #  'Nov 11, 2012 15:34:01 PM',
        #  'Alexandra Jovez',
        #  '+16095552144',
        #  "What did you end up getting?.",
        #  "iMessage"]

        timestamp = datetime.datetime.strptime(row[1], "%b %d, %Y %H:%M:%S %p")
        inbound = (row[0] == PHONEVIEW_EXTERNAL_RECEIVED)

        record = collections.OrderedDict({
            "timestamp": timestamp,
            "inbound": inbound,
            "length": len(row[4]),
            "content": row[4],
            "name": row[2],
            "number": row[3],
            "type": row[5] if len(row) >= 5 else "",
        })
        records.append(record)
    return records

emoji_dict = {}
emoji_list = []

def content_search_csv(records, keyword: str):
    for record in records:
        # searching for keywords in text
        record["keyword"] = 1 if record["content"].lower().find(keyword) != -1 else 0
        
        # finding reactions
        is_reaction = 0
        for react in ENG_REACTIONS:
            if record["content"].lower().find(react) != -1:
                is_reaction = 1
                break
        if not is_reaction:
            for react in SPA_REACTIONS:
                if record["content"].lower().find(react) != -1:
                    is_reaction = 1
                    break
        record["is_reaction"] = is_reaction
        
        # searching for emojis
        record["emoji"] = 0
        emoji_count = 0
        for c in record["content"]:
            if c in emoji.UNICODE_EMOJI:
                record["emoji"] = 1
                emoji_count += 1
                emoji_list.append(c)
                if c not in emoji_dict:
                    emoji_dict[c] = 1
                else:
                    emoji_dict[c] += 1
        record["num_emojis"] = emoji_count
    
    return records

def content_search_json(records, keyword: str):
    store_emojis = []
    for record in records:
        # searching for keywords in text
        record["keyword"] = 1 if record["content"].lower().find(keyword) != -1 else 0
        
        # finding reactions
        record["is_reaction"] = 1 if record["reactions"] else 0
        record.pop("reactions", None)
        
        # searching for emojis
        record["emoji"] = 0
        record["num_emojis"] = 0
           
    return records

def get_emoji_list(records) -> list:
    return emoji_list

def get_emoji_dict() -> dict:
    return emoji_dict

def get_most_frequent_emoji() -> str:
    champ = 0
    champ_key = 0
    for key in emoji_dict:
        if emoji_dict[key] > champ:
            champ_key = key
            champ = emoji_dict[key]
    return champ_key

def json_to_list(filepath):
    with open (filepath) as json_file:
        data = json.load(json_file)
        records = []
        if len(data["participants"]) == 2:
            sender = data["participants"][0]["name"]
            for message in data["messages"]:
                if (message["type"] == "Generic" or message["type"] == "Share") and "content" in message:
                    time_conv = datetime.datetime.fromtimestamp(int(int(message["timestamp_ms"]) / 1000))
                    record = collections.OrderedDict({
                        "timestamp": time_conv,
                        "inbound": True if message["sender_name"] == sender else False,
                        "length": len(message["content"]),
                        "content": message["content"],
                        "name": message["sender_name"],
                        "number": "",
                        "type": "",
                        "reactions": message["reactions"] if "reactions" in message else None
                    })
                    records.append(record)
            return records
        else:
            return None

def csv_to_list(filepath):
     # read CSV file using standard Python library
    with open(filepath, "r") as csv_file:
        csv_reader = csv.reader(csv_file,
                                delimiter=',',
                                quotechar='"',
                                quoting=csv.QUOTE_ALL)
        raw_rows = [row for row in csv_reader]

    if len(raw_rows) == 0:
        return None

    # if the first row contains headers, and coincides with internal
    # field names, then we have a file in the internal format
    if len(set(raw_rows[0]).intersection(set(PHONEVIEW_INTERNAL_DB_FIELDS))) > 1:

        header = raw_rows[0]

        records = _load_internal_phoneview_msg_file(
            raw_rows=raw_rows[1:],
            header=header,
        )

    else:
        records = _load_external_phoneview_msg_file(
            raw_rows=raw_rows
        )
    
    return records
        
def _normalize_phone_number(phone_number: typing.Union[str, int]) -> str:
    try:
        # using a dedicated package if available
        # see: https://github.com/daviddrysdale/python-phonenumbers
        import phonenumbers
        normalized_phone_number = phonenumbers.format_number(
            numobj=phonenumbers.parse(
                number=phone_number,
                region=DEFAULT_PHONE_REGION,
            ),
            num_format=phonenumbers.PhoneNumberFormat.E164)

    except ImportError:
        if phone_number is int:
            normalized_phone_number = str(phone_number)
        else:
            normalized_phone_number = phone_number
            for c in " ()-+":
                normalized_phone_number = normalized_phone_number.replace(c, "")

    return normalized_phone_number


# noinspection PyBroadException
def _compare_phone_numbers(
        *phone_numbers: typing.Union[str, int]
) -> bool:
    try:
        return len(set(map(_normalize_phone_number, phone_numbers))) == 1
    except:
        return False


    
# function that gets called in the jupyter notebook
def load_file(
        filepath: str,
        phone_number: typing.Optional[str] = None,
        keep_type: bool = True,
        keep_other_identity: bool = False,
        keyword: str = None
) -> pd.DataFrame:
    
    # run additional searches on records
    if (filepath[-4:] == ".csv"):
        records = csv_to_list(filepath)
        records = content_search_csv(records, keyword)
    elif (filepath[-5:] == ".json"):
        records = json_to_list(filepath)
        records = content_search_json(records, keyword)
    else:
        print("Error: Your file is named incorrectly.")
        return None
    
        
    # post process the records
    post_processed_records = []

    for record in records:

        # if user requested records for specific number, skip record if not
        # from that number
        if phone_number is not None and "number" in record:
            if not _compare_phone_numbers(record["number"], phone_number):
                continue

        # if user wants name/number dropped, remove from record
        if not keep_other_identity:
            if "name" in record:
                del record["name"]
            if "number" in record:
                del record["number"]

        if not keep_type and "type" in record:
            del record["type"]

        post_processed_records.append(record)

    if len(post_processed_records) == 0:
        return

    # sort chronologically by timestamp
    post_processed_records = sorted(
        post_processed_records,
        key=lambda record: record["timestamp"]
    )

    # create the pandas dataframe, index by timestamp and sort chronologically
    df = pd.DataFrame(post_processed_records)
    df = df.set_index("timestamp")
    df = df.sort_index(ascending=True)
    
    # Printing broad stats of dataframe
    if (filepath[-4:] == ".csv"):
        print("\nOverview of CSV file WITHOUT reactions:")
        print(df[df["is_reaction"] == False].describe())
        print("\nOverview of CSV file WITH reactions:")
        print(df.describe())
    else:
        print("\nOverview of JSON file WITH reactions:")
        print(df.describe())
    return df

def erase_labels(lst, regexp=r"([0-9]*)/[0-9]*"):

    # default regexp, saves first number
    if regexp is None:
        regexp = r"([0-9]*)/[0-9]*"

    cregexp = re.compile(regexp)

    new_lst = []
    prev_new_item = None

    for item in lst:
        new_item = ""

        m = cregexp.search(item)
        if m is not None:
            new_item = "".join(m.groups())
            if new_item == prev_new_item:
                new_item = ""
            else:
                prev_new_item = new_item

        new_lst.append(new_item)

    return new_lst


def dump_to_csv(dataframe: pd.DataFrame, filepath: str = None) -> typing.Optional[str]:
    # make sure we have the timestamp
    if dataframe.index.name is not None:
        dataframe = dataframe.reset_index()

    header = dataframe.columns.to_list()

    record_list = list()

    for row_id, row in enumerate(dataframe.values.tolist()):
        row_dict = dict(zip(header, row))

        record = collections.OrderedDict({
            "id": row_id,
            "date": int(row_dict["timestamp"].to_pydatetime().timestamp()),
            "address": row_dict.get("number", ""),
            "text": row_dict.get("content", ""),
            "flags": (
                PHONEVIEW_INTERNAL_RECEIVED if row_dict["inbound"]
                else PHONEVIEW_INTERNAL_SENT),
        })

        record_list.append(record)

    if filepath is not None:
        f = open(filepath, "w", newline="")

    else:
        f = io.StringIO(newline="")

    field_names = list(record_list[0].keys())
    writer = csv.DictWriter(
        f,
        fieldnames=field_names,
        delimiter=',',
        quotechar='"',
        quoting=csv.QUOTE_ALL)

    writer.writeheader()
    writer.writerows(record_list)

    if filepath is not None:
        return

    else:
        return f.getvalue()


def plot_texts(
        texts_df: pd.DataFrame,
        time_frequency: str = "M",
        split_by_direction: bool = True,
        rescaled: bool = False,
        remove_gaps: bool = False,
        setting: PlotStyle = PlotStyle.COUNT,
        label_date_format: typing.Optional[str] = None,
        colors=None,
        date_start: typing.Optional[str] = None,
        date_end: typing.Optional[str] = None,
        remove_reactions: bool = True,
        show_frequency: bool = False,
):

    df = texts_df
    
    def aggregate_df(df):
        if setting == PlotStyle.COUNT:
            agg_df = df.resample(time_frequency).count().drop(columns=["length", "content", "keyword", "is_reaction", "emoji", "num_emojis"])
        elif setting == PlotStyle.VOLUME:
            agg_df = df.resample(time_frequency).sum().drop(columns=["inbound", "keyword", "is_reaction", "emoji", "num_emojis"])
        elif setting == PlotStyle.KEYWORD:
            if show_frequency:
                agg_df = df.resample(time_frequency).mean().drop(columns=["length", "inbound", "is_reaction", "emoji", "num_emojis"])
            else:
                agg_df = df.resample(time_frequency).sum().drop(columns=["length", "inbound", "is_reaction", "emoji", "num_emojis"])
        elif setting == PlotStyle.REACTIONS:
            if show_frequency:
                agg_df = df.resample(time_frequency).mean().drop(columns=["length", "inbound", "keyword", "emoji", "num_emojis"])
            else:
                agg_df = df.resample(time_frequency).sum().drop(columns=["length", "inbound", "keyword", "emoji", "num_emojis"])
        else:
            raise ValueError("{} is not a valid value for `setting`".format(
                setting,
            ))

        column_name = agg_df.columns.to_list()[0]
        if remove_gaps:
            agg_df = agg_df[agg_df[column_name] > 0]
        return agg_df

    if split_by_direction:
        
        # split by inbound/outbound
        df_inbound = df[df["inbound"] == True] # received
        df_outbound = df[df["inbound"] == False] # sent
        
        if remove_reactions:
            df_inbound = df_inbound[df_inbound["is_reaction"] == False]
            df_outbound = df_outbound[df_outbound["is_reaction"] == False]
        
        # stack the two datasets
        processed_df = pd.merge(
                aggregate_df(df_inbound),
                aggregate_df(df_outbound),
                how="outer",
                on=["timestamp"]
            )

        # rename columns
        processed_df.rename(columns={"inbound_x": "received", "inbound_y": "sent"}, inplace=True)
        processed_df.rename(columns={"length_x": "received", "length_y": "sent"}, inplace=True)
        processed_df.rename(columns={"keyword_x": "received", "keyword_y": "sent"}, inplace=True)  
        processed_df.rename(columns={"is_reaction_x": "received", "is_reaction_y": "sent"}, inplace=True)
        processed_df.rename(columns={"num_emojis_x": "received", "num_emojis_y": "sent"}, inplace=True)        

#         print(processed_df.describe())

    else:
        processed_df = aggregate_df(df)

    if rescaled:
        processed_df = processed_df.div(processed_df.sum(1), axis=0)

    # compute the title based on the parameters
    title = "Rescaled " if rescaled else "Absolute "
    if setting == PlotStyle.COUNT:
        title += "Number of Messages"
    elif setting == PlotStyle.VOLUME:
        title += "Volume (in characters) of Messages"
    elif setting == PlotStyle.KEYWORD:
        if show_frequency:
            title += "Percentage of Messages Containing Keyword"
        else:
            title += "Number of Messages Containing Keyword"
    elif setting == PlotStyle.REACTIONS:
        if show_frequency:
            title += "Percentage of Reactions"
        else:
            title += "Number of Reactions"

    if split_by_direction:
        title += " (by direction)"
    if remove_reactions:
        title += " (excluding reactions)"

    # if we wanted to cut the range
    if date_start is not None or date_end is not None:
        processed_df = processed_df.loc[date_start:date_end]

    # actually plot this data
    ax = processed_df.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 5),
        title=title,
        color=colors,
    )

    _ = ax.set_xlabel("Date")
    _ = ax.set_xticklabels(
        erase_labels([
            pandas_datetime.strftime("%Y/%m/%d %H:%m")
            for pandas_datetime in processed_df.index
        ], regexp=label_date_format))

    return ax.get_figure()
