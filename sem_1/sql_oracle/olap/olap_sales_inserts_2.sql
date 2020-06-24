create sequence SEQ_DIM_COUNTRY_ID;

select SEQ_DIM_COUNTRY_ID.nextval from dual;

insert into DIM_COUNTRY(COUNTRY_ID, COUNTRY_NAME)
select SEQ_DIM_COUNTRY_ID.nextval, C.country
from (select distinct COUNTRY from CITY) C;

insert into DIM_COUNTRY(COUNTRY_ID, COUNTRY_NAME)
values (-1, 'N/A');

select * from DIM_COUNTRY;

select distinct country from city;

insert into DIM_CITY(CITY_ID, CITY_NAME, COUNTRY_ID)
select CITY_ID, CITY_NAME,
(select COUNTRY_ID from DIM_COUNTRY where COUNTRY_NAME = CITY.COUNTRY)
from CITY;

insert into DIM_CITY(CITY_ID, CITY_NAME, COUNTRY_ID)
values (-1, 'N/A', -1);

select * from DIM_CITY;

insert into DIM_OFFICE(OFFICE_ID,OFFICE_NAME,CITY_ID)
select OFFICE_ID, OFFICE_NAME, CITY_ID
from OFFICE;

insert into DIM_OFFICE(OFFICE_ID,OFFICE_NAME,CITY_ID)
values(-1, 'N/A', -1);

insert into DIM_MANAGER(MANAGER_ID,FIRST_NAME,LAST_NAME)
select MANAGER_ID, MANAGER_FIRST_NAME, MANAGER_LAST_NAME from MANAGER;

insert into DIM_MANAGER(MANAGER_ID,FIRST_NAME,LAST_NAME)
values (-1, NULL, 'N/A');

insert into DIM_PRODUCT(PRODUCT_ID,PRODUCT_NAME)
select PRODUCT_ID, PRODUCT_NAME from PRODUCT;
commit;

select MIN(ORDER_DATE), MAX(ORDER_DATE) from SALES_ORDER;

begin
 P_GENERATE_DIM_DATE(
  TO_DATE('01.01.2000', 'DD.MM.YYYY'),
  TO_DATE('31.12.2016', 'DD.MM.YYYY')
);
end;

rollback;
delete from DIM_DATE;
commit;

select * from DIM_DATE;

create table STG_SALES as
select * from V_FACT_SALE;

select * from STG_SALES;

insert into FACT_SALES(
SALE_ID,
PRODUCT_ID,
MANAGER_ID,
COUNTRY_ID,
CITY_ID,
OFFICE_ID,
DATE_ID,
SALE_AMOUNT,
SALE_PRICE,
SALE_QTY)
select
S.FACT_ID,
S.PRODUCT_ID,
NVL(S.MANAGER_ID, -1),
NVL((select COUNTRY_ID from DIM_COUNTRY where COUNTRY_NAME = S.COUNTRY), -1),
NVL(S.CITY_ID, -1),
NVL(S.OFFICE_ID, -1),
(select DATE_ID from DIM_DATE where DATE_NAME = TO_CHAR(S.SALE_DATE, 'DD.MM.YYYY')),
S.SALE_AMOUNT,
S.SALE_PRICE,
S.SALE_QTY
from STG_SALES S;
commit;

select count(*) from SALES_ORDER_LINE;

select * from FACT_SALES;

select * from SALES_ORDER where MANAGER_ID IS NULL;

select MANAGER_ID, PRODUCT_ID, SUM(SALE_AMOUNT), AVG(SALE_PRICE), SUM(SALE_QTY) from FACT_SALES
group by MANAGER_ID, PRODUCT_ID
union all
select MANAGER_ID, NULL, SUM(SALE_AMOUNT), AVG(SALE_PRICE), SUM(SALE_QTY) from FACT_SALES
group by MANAGER_ID
union all
select NULL, PRODUCT_ID, SUM(SALE_AMOUNT), AVG(SALE_PRICE), SUM(SALE_QTY) from FACT_SALES
group by PRODUCT_ID
union all
select NULL, NULL, SUM(SALE_AMOUNT), AVG(SALE_PRICE), SUM(SALE_QTY) from FACT_SALES
;

drop table FACT_SALES_AGG;
create table FACT_SALES_AGG as select * from FACT_SALES where ROWNUM < 1;

insert into FACT_SALES_AGG(
PRODUCT_ID,
MANAGER_ID,
COUNTRY_ID,
CITY_ID,
OFFICE_ID,
DATE_ID,
SALE_AMOUNT,
SALE_PRICE,
SALE_QTY
)
select PRODUCT_ID, MANAGER_ID, COUNTRY_ID, CITY_ID, OFFICE_ID, DATE_ID,
SUM(SALE_AMOUNT), AVG(SALE_PRICE), SUM(SALE_QTY)
from FACT_SALES
group by CUBE(MANAGER_ID, PRODUCT_ID, DATE_ID), ROLLUP(COUNTRY_ID, CITY_ID, OFFICE_ID)
having OFFICE_ID IS NULL and CITY_ID IS NOT NULL;

drop materialized view MV_FACT_SALES_AGG;

create materialized view MV_FACT_SALES_AGG as
select PRODUCT_ID, MANAGER_ID, COUNTRY_ID, CITY_ID, OFFICE_ID, DATE_ID,
SUM(SALE_AMOUNT), AVG(SALE_PRICE), SUM(SALE_QTY)
from FACT_SALES
group by CUBE(MANAGER_ID, PRODUCT_ID, DATE_ID), ROLLUP(COUNTRY_ID, CITY_ID, OFFICE_ID)
having OFFICE_ID IS NULL and CITY_ID IS NOT NULL;

select * from MV_FACT_SALES_AGG;

commit;

select * from FACT_SALES_AGG;

drop materialized view MV_RAW_FACT_SALES;

create materialized view MV_RAW_FACT_SALES REFRESH FAST as
select ol.product_qty, ol.product_price, ol.product_id, o.manager_id,
    ofc.city_id,
    ol.order_line_id
from SALES_ORDER_LINE ol
   inner join SALES_ORDER o on (ol.sales_order_id = o.sales_order_id)
   inner join MANAGER@ORCL_MAI m on (o.manager_id = m.manager_id)
   inner join OFFICE@ORCL_MAI ofc on (ofc.office_id = m.office_id)
   inner join CITY@ORCL_MAI c on (ofc.city_id = c.city_id);

select * from ALL_OBJECTS where OBJECT_NAME like 'MV_RAW_FACT_SALES';

select count(*) from MV_RAW_FACT_SALES; -- 54944 / 54944 / 54945
select count(*) from V_RAW_FACT_SALES; -- 54944 / 54945

begin
   dbms_mview.refresh('MV_FACT_SALES_AGG');
end;

create materialized view MV_COMPANY_2 REFRESH FAST
       START WITH SYSDATE NEXT TRUNC(SYSDATE+1) + 3/24 as
       select * from COMPANY;

create materialized view log on COMPANY
       with primary key
       including new values;

