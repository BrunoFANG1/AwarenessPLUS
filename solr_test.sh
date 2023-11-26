bin/solr start

bin/solr create -c test -s 2 -rf 2

curl -X POST -H 'Content-type:application/json' --data-binary '{
"add-field": {
"name": "data_publish",
"type": "text_general",
"multiValued": false,
"stored": true,
"indexed": true
},

"add-field": {
"name": "outlet",
"type": "text_general",
"multiValued": false,
"stored": true,
"indexed": true
},

"add-field": {
"name": "headline",
"type": "text_general",
"multiValued": false,
"stored": true,
"indexed": true
},

"add-field": {
"name": "lead",
"type": "text_general",
"multiValued": false,
"stored": true,
"indexed": true
},

"add-field": {
"name": "body",
"type": "text_general",
"multiValued": false,
"stored": true,
"indexed": true
},

"add-field": {
"name": "authors",
"type": "text_general",
"multiValued": false,
"stored": true,
"indexed": true
},
}' [http://localhost:8983/solr/test/schema](http://localhost:8983/solr/your_core_name/schema)

# Search "Trump" in all body
curl "[http://localhost:8983/solr/test/select?q=body:on Monday](http://localhost:8983/solr/test/select?q=body:Trump)"