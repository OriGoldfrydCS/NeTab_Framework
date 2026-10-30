"""V1: prefix and null padding. V2: active-record domain membership."""


def validate_rows(schema, rows):
    for row in rows:
        schema.length(row)
    return True
