from neo4j import GraphDatabase
import pandas as pd
import networkx as nx

print("Read CSV")
data = pd.read_csv('./data/2023.csv')
# G = nx.from_pandas_edgelist(data, 'STARTSTATIONNAME', 'ENDSTATIONNAME')

# Define Neo4j connection parameters
uri = "neo4j://localhost:7687"

# Function to import nodes
def add_cycles(driver, nodes):
    # for node_data in nodes:
    #     tx.run("CREATE (n:Node {id: $id, prop1: $prop1, prop2: $prop2})", **node_data)
    
    for index, row in nodes.iterrows():
        #print(row)
        if row.isna().any():
            print(row)
            continue
        
        driver.execute_query(
            "MERGE (n:Station {name: $STARTSTATIONNAME, latitude: $STARTSTATIONLATITUDE, longitude: $STARTSTATIONLONGITUDE}) " \
            "MERGE (m:Station {name: $ENDSTATIONNAME, latitude: $ENDSTATIONLATITUDE, longitude: $ENDSTATIONLONGITUDE}) " \
            "MERGE (n)-[:CYCLES_TO {starttime: $STARTTIMEMS, endtime: $ENDTIMEMS}]->(m)",
            STARTSTATIONNAME = row['STARTSTATIONNAME'],
            STARTSTATIONLATITUDE = row['STARTSTATIONLATITUDE'],
            STARTSTATIONLONGITUDE = row['STARTSTATIONLONGITUDE'],
            ENDSTATIONNAME = row['ENDSTATIONNAME'],
            ENDSTATIONLATITUDE = row['ENDSTATIONLATITUDE'],
            ENDSTATIONLONGITUDE = row['ENDSTATIONLONGITUDE'],
            STARTTIMEMS = row['STARTTIMEMS'],
            ENDTIMEMS = row['ENDTIMEMS']
        )

print("Write data to Neo4j")
# Create session and import data
with GraphDatabase.driver(uri) as driver:
    # session.execute_write(import_nodes, nodes)
    # session.execute_write(import_relationships, relationships)
    add_cycles(driver, data)

# Close the driver
driver.close()