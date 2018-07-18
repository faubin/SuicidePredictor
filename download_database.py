import os
import time
import requests
import pycurl


def create_data_structure():
    """
    ensure a downloads folder exist and returns the list of files in it
    """
    if 'downloads' not in os.listdir('.'):
        os.mkdir('downloads')
        file_list = []
    else:
        file_list = os.listdir('downloads')
    return file_list


def clear_data_structure():
    """
    erase the downloads folder to start fresh
    """
    if 'downloads' in os.listdir('.'):
        os.system('rm -r downloads')
    return


def guess_names(file_name):
    """
    return a list of potential file names to download
    """
    name = file_name[:-9]

    possible_names = []
    # for v in range(9, 0, -1):
    for v in range(6, 0, -1):
        for ext in ['', '_0', '_1']:
            for pre in ['%20', '']:
                for rank in ['Rankings', 'Ranking']:
                    new_name = '{0:s}{1:s}v{2:d}{3:s}.xls'.format(name, pre, v,
                                                                  ext)
                    if not rank == 'Rankings':
                        new_name = new_name.replace('Rankings', rank)
                    possible_names.append(new_name)
    return possible_names


def download_file(base_url, file_name, save_file_name, file_size_lowerlim=0,
                  sleep_time=3):
    """
    download a file, save it as save_file_name, erase it if it is smaller than
    file_size_lower_limit (in Bytes)
    """
    url = os.path.join(base_url, file_name)
    print(file_name)
    print('Downloading {0:s}'.format(save_file_name))

    file_ = open(os.path.join('downloads', save_file_name), 'wb')
    curl = pycurl.Curl()
    curl.setopt(pycurl.URL, url)
    curl.setopt(pycurl.FOLLOWLOCATION, 1)
    curl.setopt(pycurl.MAXREDIRS, 5)
    curl.setopt(pycurl.CONNECTTIMEOUT, 30)
    curl.setopt(pycurl.TIMEOUT, 300)
    curl.setopt(pycurl.NOSIGNAL, 1)
    curl.setopt(pycurl.WRITEDATA, file_)
    curl.perform()
    file_.close()
    time.sleep(sleep_time)

    file_size = os.path.getsize(os.path.join('downloads', save_file_name))
    if file_size < file_size_lowerlim:
        print('File size is {0:d} B'.format(file_size))
        os.system("rm downloads/{0:s}".format(save_file_name.replace(' ',
                                                                     '/ ')))
        return False
    else:
        f = open("downloaded.txt", 'a')
        f.write(file_name + '\n')
        f.close()
        return True


sleep_time = 3
downloaded_files = create_data_structure()

base_url = os.path.join("http://www.countyhealthrankings.org", "sites",
                        "default", "files", "state", "downloads")

states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
          'Connecticut', 'Delaware', 'District of Columbia', 'Florida',
          'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
          'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
          'Massachusetts', 'Michigan', 'Minnesota', 'Missouri', 'Montana',
          'Nebraska', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
          'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
          'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
          'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
          'West Virginia', 'Wisconsin', 'Wyoming']

for year in [2005, 2014, 2015, 2016]:
    base_url = 'https://www.cdc.gov/nchs/pressroom/sosmap/suicide-mortality/'
    url_file_name = 'SUICIDE{0:d}.csv'.format(year)
    success = download_file(base_url, url_file_name, url_file_name,
                            file_size_lowerlim=1000, sleep_time=sleep_time)
    if not success:
        exit("Error downloading suicide data")

years = range(2010, 2019)
for state in states:
    for year in years:
        print state, year
        file_name = "{0:d}%20County%20Health%20Rankings".format(year)
        url_state = state.replace(' ',   '%20')
        file_name += "%20{0:s}%20Data%20-%20v1.xls".format(url_state)

        save_file_name = '{0:s}_{1:d}.xls'.format(state, year)

        if save_file_name not in downloaded_files:
            print '---'
            possible_url_file_names = guess_names(file_name)
            for url_file_name in possible_url_file_names:
                success = download_file(base_url, url_file_name,
                                        save_file_name,
                                        file_size_lowerlim=29000,
                                        sleep_time=sleep_time)
                if success:
                    break
