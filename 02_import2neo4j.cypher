// https://neo4j.com/docs/getting-started/data-import/csv-import/#optimizing-load-csv

// LOAD CSV USING CALL AND COALESCE
// the file 2023.csv should be place in the import folder on the server! (see docker run command)
:auto LOAD CSV WITH HEADERS
FROM 'file:///2023.csv' AS row
CALL {
 WITH row
 MERGE (s1:Station {
     name: COALESCE(row.STARTSTATIONNAME, 'NA_STARTSTATIONNAME'), 
     latitude: COALESCE(row.STARTSTATIONLATITUDE, 'NA_STARTSTATIONLATITUDE'), 
     longitude: COALESCE(row.STARTSTATIONLONGITUDE, 'NA_STARTSTATIONLONGITUDE')
     })
 MERGE (s2:Station {
     name: COALESCE(row.ENDSTATIONNAME, 'NA_ENDSTATIONNAME'), 
     latitude: COALESCE(row.ENDSTATIONLATITUDE, 'NA_ENDSTATIONLATITUDE'), 
     longitude: COALESCE(row.ENDSTATIONLONGITUDE, 'NA_ENDSTATIONLONGITUDE')
     })
 MERGE (s1)-[r:CYCLES_TO {
     starttime: COALESCE(row.STARTTIMEMS, 'NA_STARTTIMEMS'), 
     endtime: COALESCE(row.ENDTIMEMS, 'NA_ENDTIMEMS')
     }]->(s2)
} IN TRANSACTIONS OF 100000 ROWS;

//Delete everything
MATCH ()-[r]-()
DELETE r;

// Delete all nodes
MATCH (n)
DELETE n;
