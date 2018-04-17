import mechanize
import cookielib
# from BeautifulSoup import BeautifulSoup
# import html2text

# Browser
br = mechanize.Browser()

# Cookie Jar
cj = cookielib.LWPCookieJar()
br.set_cookiejar(cj)

# Browser options
br.set_handle_equiv(True)
br.set_handle_gzip(True)
br.set_handle_redirect(True)
br.set_handle_referer(True)
br.set_handle_robots(False)

# Follows refresh 0 but not hangs on refresh > 0
br.set_handle_refresh(mechanize._http.HTTPRefreshProcessor(), max_time=1)

# User-Agent (this is cheating, ok?)
br.addheaders = [('User-agent', 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1')]

# The site we will navigate into, handling it's session
br.open('http://www.cpcb.gov.in/CAAQM/frmUserAvgReportCriteria.aspx')

# Select the first (index zero) form
br.select_form(nr=0)

# User credentials
br.form['ddlState'] = ['6']
print("Getting cities..")
br.submit()
br.select_form(nr=0)
br.form['ddlCity'] = ['85']
br.submit()
br.select_form(nr=0)
print("Getting Stations..")
br.form['ddlStation'] = ['46']
br.submit()
br.select_form(nr=0)

br.form['lstBoxChannelLeft'] = ['460']
br.submit(name='btnAdd')
br.select_form(nr=0)
# print(br.form)
br.form['ddlCriteria'] = ['1']
br.form['txtDateFrom'] = '01/01/2016'
br.form['txtDateTo'] = '29/02/2016'
response = br.submit(name='btnSubmit')
print(response.code)




# Login
