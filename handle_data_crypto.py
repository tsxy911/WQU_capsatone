import taosws
import pandas as pd
import time
from itertools import product
import sys

def querry_table(auth, db, table_type):
    now_table = []
    conn = taosws.connect(auth+'/'+db)
    sql = "SHOW "+table_type
    result = conn.query(sql)
    for row in result:
        now_table.append(row[0])
    conn.close()
    return now_table


# 超级表SQL语句生成器
def generate_create_stable_sql(df, stable_name, tags):
    # 检查 TAGS 中的每个列名是否在 DataFrame 中存在
    for tag in tags:
        if tag not in df.columns:
            raise ValueError(f"The column '{tag}' specified in TAGS does not exist in the DataFrame.")
    df_columns = df.columns
    nor_columns = [i for i in df.columns if i not in tags]
    df_dtypes = df.dtypes
    
    create_table_sql = 'CREATE STABLE ' + stable_name + ' ('

    for column in nor_columns:
        sql_type = str(df_dtypes[column]).replace('dtype', '')  # 转换 pandas 类型为字符串
        tdengine_type = {
            'int64': 'INT',
            'float64': 'DOUBLE',
            'object': 'NCHAR(255)',
            'bool': 'BOOL',
            'datetime64[ns]': 'TIMESTAMP'
        }.get(sql_type, 'NCHAR(255)')  # 默认为 NHAR(255)
        
        create_table_sql += f"{column} {tdengine_type},"
        
    # 移除最后的逗号
    create_table_sql = create_table_sql.rstrip(',')
    
    # 添加结束括号和存储引擎
    create_table_sql += ')'
    
    create_table_sql += f"TAGS ("
    
    for column in tags:
        sql_type = str(df_dtypes[column]).replace('dtype', '')  # 转换 pandas 类型为字符串
        tdengine_type = {
            'int64': 'INT',
            'float64': 'DOUBLE',
            'object': 'NCHAR(255)',
            'bool': 'BOOL',
            'datetime64[ns]': 'TIMESTAMP'
        }.get(sql_type, 'NCHAR(255)')  # 默认为 NHAR(255)
        
        create_table_sql += f"{column} {tdengine_type},"
    
    # 移除最后的逗号
    create_table_sql = create_table_sql.rstrip(',')
    
    # 添加结束括号和存储引擎
    create_table_sql += ');'
    # print(create_table_sql)
    return create_table_sql


def create_db(auth, db, VGROUPS, KEEP):
    now_db = []
    conn = taosws.connect(auth)
    sql = "create database if not exists {} VGROUPS {} PRECISION 'ms' KEEP {}d"
    sql = sql.format(db, VGROUPS)
    conn.execute(sql)
    result = conn.query("SHOW DATABASES")
    for row in result:
        now_db.append(row[0])
    print('新建数据库完成，目前存在数据库：', now_db)
    conn.close()
    
# 创建超级表
def create_st(data, auth, db, stable_name, tags):
    table_type = 'STABLES'
    check = querry_table(auth, db, table_type)
    if stable_name in check:
        print(stable_name, '已存在，跳过新建超级表')
    else:
        now_stable = []
        sql = generate_create_stable_sql(data, stable_name, tags)
        conn = taosws.connect(auth+'/'+db)
        conn.execute(sql)
        result = conn.query("SHOW STABLES")
        for row in result:
            now_stable.append(row[0])
        conn.close()
        if stable_name in now_stable:
            print('创建超级表', stable_name, '执行完成')
            
# 创建子表
def create_tb(data, auth, db, stable_name, tags, batch_size):
    table_type = 'TABLES'
    check = querry_table(auth, db, table_type)
    # print('check')
    
    tag_values = []
    conn = taosws.connect(auth+'/'+db)
    for columns in tags:
        unique_values = pd.Series(data[columns].unique())
        if unique_values.dtype == 'float' or unique_values.dtype == 'int':
            pass
        else:
            unique_values = unique_values.str.replace('.', '')
        tag_values.append(list(unique_values))
    tb_head = stable_name + '_'
    # tb_names = ['_'.join(str(item) for item in combination) for combination in itertools.product(*tag_values)]
    tb_names = ['_'.join(item) for item in product(*tag_values)]
    # print(tb_names)
    for t in range(len(tb_names)):
        tb_name = tb_names[t]
        tb = tb_head + tb_name
        tb = tb.lower()
        if tb not in check:
            status = '新建完成'
            tag_v = tb_name.split('_')
            sql = "CREATE TABLE " + tb + " USING " + stable_name + " TAGS " + str(tuple(tag_v))
            # print(sql)
            conn.execute(sql)
        else:
            tag_v = tb_name.split('_')
            status = '已存在，跳过建表'
        # 根据tags筛选数据
        conditions = [data[tag] == value for tag, value in zip(tags, tag_v)]
        combined_condition = conditions[0]
        for condition in conditions[1:]:
            combined_condition &= condition  # 逐个添加条件并进行逻辑与操作

        # 使用组合条件筛选 DataFrame
        filtered_data = data[combined_condition]
        filtered_data = filtered_data.reset_index(drop = True)

        filter_dict = dict(zip(tags, tag_v)) 
            
        for start in range(0, len(filtered_data), batch_size):
            end = min(start + batch_size, len(filtered_data))
            chunk = filtered_data.iloc[start:end]
            chunk = chunk.reset_index(drop = True)
            res = write_df_to_td(chunk, auth, db, stable_name, tb, filter_dict)
            sys.stdout.write('\r'+ str(t) + '/' + str(len(tb_names)) + ':子表' + str(tb) + status + str(start) + '--' + str(end))
    print('落库完成')
    conn.close()
    
    
# 写入数据
def write_df_to_td(df, auth, db, stable, table, tags):
    conn = taosws.connect(auth+'/'+db)
    cols = {}
    tcols = {}
    res = conn.query("describe {}".format(stable))
    for row in res:
        if row[3] == 'TAG':
            tcols[row[0]] = row[1]
        else:
            cols[row[0]] = row[1]
    
    # 超级表名
    def parse_stable(stable):
        return stable

    # 子表名
    def parse_table(table):
        return table

    # 标签值，不同表有不同的标签值，在这里可以进行扩展
    def parse_tags(tags):
        s = []
        for t in tcols.keys():
            if t in tags.keys():
                if tcols[t] in ["TIMESTAMP","NCHAR","VARCHAR"]:
                    s.append("'{}'".format(tags[t]))
                else:
                    s.append("{}".format(tags[t]))
        return ','.join(s)
    
    tag_keys = list(tags.keys())
    if tag_keys:
        df = df.drop(columns=tag_keys)
        
    df_cols = df.columns.tolist()
    df[df_cols[0]] = pd.to_datetime(df[df_cols[0]])
    df[df_cols[0]] = df[df_cols[0]].dt.strftime('%Y-%m-%d %H:%M:%S.%f')

    sql = "INSERT INTO {}({}) USING {} TAGS({}) VALUES ".format(parse_table(table), ",".join(df_cols), parse_stable(stable), parse_tags(tags))
    for line in df.values.tolist():
        sql += "("+"{}".format(line)[1:-1]+")"
    sql = sql.replace(", None", ", NULL")
    sql = sql.replace(", nan", ", NULL")
    # print(sql)
    # sys.stdout.write('\r'+ str(df) + str(sql))
    return conn.execute(sql)


def td_querry(sql, auth, db):
    start_time = time.time() 
    conn = taosws.connect(auth+'/'+db)
    result = conn.query(sql)
    columns = [x.name() for x in result.fields]
    data = pd.DataFrame([d for d in result], columns=columns)

    end_time = time.time()  # 结束计时
    # print(f"执行时间：{end_time - start_time} 秒")
    return data

def deal_data(data):
    data['time'] = pd.to_datetime(data['time'])
    data.columns = data.columns.str.lower()
    return data

def collect_data(data, auth, create_new_db, create_new_st, VGROUPS, KEEP, db, stable_name, tags, batch_size):
    if create_new_db == True:
        create_db(auth, db, VGROUPS, KEEP)
    
    if create_new_st == True:
        create_st(data, auth, db, stable_name, tags)
    elif create_new_st == False:
        print('不创建新的超级表，在超级表',stable_name,'中创建子表')
        
    create_tb(data, auth, db, stable_name, tags, batch_size)