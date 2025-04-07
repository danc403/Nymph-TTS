#!/usr/bin/env python3

import pyarrow as pa
import pyarrow.flight as fl
import pyarrow.parquet as pq
import os

class ParquetFlightServer(fl.FlightServerBase):
    def __init__(self, location, data_dir):
        super(ParquetFlightServer, self).__init__(location)
        self.data_dir = data_dir

    def do_get(self, context, ticket):
        file_list = ticket.ticket.decode().split(",")  # Split filenames from ticket
        tables = []
        for filename in file_list:
            file_path = os.path.join(self.data_dir, filename)
            try:
                table = pq.read_table(file_path)
                tables.append(table)
            except FileNotFoundError:
                print(f"File not found: {filename}")
        if tables:
            combined_table = pa.concat_tables(tables)
            return fl.RecordBatchStream(combined_table)
        else:
            return fl.RecordBatchStream(pa.Table.from_pandas(pa.pandas_api.empty_table(pa.schema([])))) #return empty table.

def run_flight_server(data_dir, host="0.0.0.0", port=9090):
    location = fl.Location.for_grpc_tcp(host, port)
    server = ParquetFlightServer(location, data_dir)
    server.serve()

if __name__ == "__main__":
    data_dir = "/models/datasets/" #change path.
    run_flight_server(data_dir)
