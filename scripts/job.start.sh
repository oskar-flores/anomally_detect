readonly PROJECT_NAME='anomlayDetector'
readonly DRIVER_CORE=1
readonly files='--files /usr/hdp/current/spark-client/conf/hive-site.xml'
readonly jars='--jars /usr/hdp/current/spark-client/lib/datanucleus-api-jdo-3.2.6.jar,/usr/hdp/current/spark-client/lib/datanucleus-rdbms-3.2.9.jar,/usr/hdp/current/spark-client/lib/datanucleus-core-3.2.10.jar'
readonly conf='--conf spark.yarn.executor.memoryOverhead=2048'

#. env.sh
spark-submit --verbose \
    --master yarn \
    --deploy-mode cluster \
    ${jars} \
    ${files} \
    ${conf} \
    ${@}
