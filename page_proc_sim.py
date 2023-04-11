import logging
import os
from os import path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# The selector of paging algorithm
G_PAGING_ALG = 'local'
# The number of processes in the simulation
G_N_PROC = 3
# The min number of pages needed per process
G_MIN_N_PAGES_PER_PROC = 3
# The max number of pages needed per process
G_MAX_N_PAGES_PER_PROC = 5
# The min lifetime of process
G_MIN_LIFE = 20
# The max lifetime of process
G_MAX_LIFE = 50
# The max number of pages in memory
G_MAX_N_PAGES_IN_MEM = 5
# The folder to take outputs
G_OUT_FOLDER = './out/'
if not path.exists(G_OUT_FOLDER):
    os.mkdir(G_OUT_FOLDER)

# Process states
# 1. Ready to be scheduled yet not running
G_PROC_READY = 1
# 2. Running now
G_PROC_RUN = 2
# 3. Not ready to be scheduled
G_PROC_WAIT = 3
# 4. Terminated
G_PROC_TERM = 4

class PageProcSim:
    # Current time
    m_time = 0
    # Global Process ID
    m_g_pid = 0
    # Alive process data (pandas DataFrame)
    m_df_proc = None
    # Currently scheduled process's ID
    m_cur_pid = None
    # Global Page ID
    m_g_page_id = 0
    # If a page fault happens at the current moment
    m_page_fault = 0
    # Page data (pandas DataFrame)
    m_df_page = None
    # Max number of in-memory pages
    m_max_n_pages_in_mem = 0
    # Collected data for analysis
    m_l_snapshots = []

    def __init__(self, n_proc, min_n_pages_per_proc, max_n_pages_per_proc, min_life, max_life, max_n_pages_in_mem):
        self.m_max_n_pages_in_mem = max_n_pages_in_mem
        self.gen_proc(n_proc, min_n_pages_per_proc, max_n_pages_per_proc, min_life, max_life)
        self.init_page_data()

    def snapshot(self):
        """
        Record all necessary data for the current moment.
        :return: No explict returned value.
        """
        # TODO
        #   Feel free to modify this function to record any necessary data you need.
        self.m_l_snapshots.append((self.m_time, self.m_page_fault, self.m_df_proc, self.m_df_page))

    def output_snapshots(self):
        # TODO
        #   If you have modified the function 'snapshot()', you may also need to modify this function for output.
        df_snapshots = pd.DataFrame(self.m_l_snapshots, columns=['time', 'pf', 'df_proc', 'df_page'])
        df_snapshots = df_snapshots.set_index('time')
        out_path = path.join(G_OUT_FOLDER, 'ana_data_%s.pickle' % datetime.now().strftime('%Y%M%d%H%M%S'))
        pd.to_pickle(df_snapshots, out_path)
        logging.debug('[PageProcSim:output_snapshots] Output: out_path')

    def start(self):
        while True:
            if path.exists('./SHUTDOWN') or (self.m_df_proc['state'] == G_PROC_TERM).all():
                break
            self.proc_sched()
            ref_page_id, ro = self.gen_page_ref()
            page_fault = self.page_access(ref_page_id, ro)
            self.m_page_fault = page_fault
            self.snapshot()
            self.clock_interrupt()
            self.m_time += 1
        self.output_snapshots()

    def page_access(self, ref_page_id, ro=True):
        """
        Access the required page. Determine if a page fault is about to occur.
        And, if so, determine which pages are about to be evicted.
        :param cur_proc_data: The profile of the currently scheduled process.
            pandas DataFrame
        :param ref_page_id: The currently referenced page ID.
            int
        :param ro: Indicates the reference is read-only.
            boolean
            - True: Read only
            - False: Read and Write
        :return: Whether a page fault occurs.
            boolean
            - True: Page fault occurs.
            - False: No page fault.
        """
        self.m_df_proc.loc[self.m_cur_pid, 'page_ref_hist'].append((self.m_time, ref_page_id))

        if self.m_df_page.loc[ref_page_id]['in_mem']:
            return False

        if len(self.m_df_page[self.m_df_page['in_mem'] == True]) >= self.m_max_n_pages_in_mem:
            page_fault = True
            self.paging(ref_page_id)
        else:
            # CAUTION
            #   To avoid the "SettingWithCopyWarning" exception, it's necessary to put 'ref_page_id' and 'in_mem' both into 'loc[]'.
            self.m_df_page.loc[ref_page_id, 'in_mem'] = True
            page_fault = False

        self.m_df_page.loc[ref_page_id, 'R'] = 1
        if not ro:
            self.m_df_page.loc[ref_page_id, 'M'] = 1

        return page_fault

    def paging(self, ref_page_id):
        """
        When a page fault occurs, select one or multiple in-memory pages to evict.
        The corresponding page data is updated.
        :param cur_proc_data: The information of the currently scheduled process.
            pandas Series
            not None
        :param ref_page_id: The page ID referenced by the currently scheduled process.
            int
        :return: No explicit returned value.
        """
        # TODO
        #   Implement paging algorithms.
        if G_PAGING_ALG == 'local':
            l_in_mem_page_ids = [page_id for page_id in self.m_df_proc.loc[self.m_cur_pid]['l_page_ids']
                                 if page_id != ref_page_id and self.m_df_page.loc[page_id]['in_mem']]
            out_page_id = np.random.choice(l_in_mem_page_ids)
            self.m_df_page[out_page_id, 'in_mem'] = False

    def clock_interrupt(self):
        # Clear the 'R' bit for all pages.
        self.m_df_page['R'] = False

    def gen_proc(self, n_proc, min_n_pages_per_proc, max_n_pages_per_proc, min_life, max_life):
        l_proc = []
        for i in range(n_proc):
            pid = self.m_g_pid
            n_pages = np.random.randint(low=min_n_pages_per_proc, high=max_n_pages_per_proc+1)
            l_page_ids = list(range(self.m_g_page_id, self.m_g_page_id + n_pages))
            page_trans_mat = self.gen_page_trans_mat_right(len(l_page_ids))
            page_ref_hist = []
            life = np.random.randint(low=min_life, high=max_life)
            birth = self.m_time
            state = G_PROC_READY
            state_hist = []
            l_proc.append((pid, state, life, birth, n_pages, l_page_ids, page_trans_mat, page_ref_hist, state_hist))
            self.m_g_pid += 1
            self.m_g_page_id += n_pages
        l_cols = ['pid', 'state', 'life', 'birth', 'n_pages', 'l_page_ids', 'page_trans_mat', 'page_ref_hist', 'state_hist']
        self.m_df_proc = pd.DataFrame(l_proc, columns=l_cols)
        self.m_df_proc = self.m_df_proc.set_index('pid')
        logging.debug('[PageProcSim:gen_proc] Generated %s processes.' % len(self.m_df_proc))

    def proc_sched(self):
        """
        Select one process to run for the current moment. And, update all involved processes.
        :return: No explicit returned value.
        """
        # TODO
        #   Needs other scheduling algorithms.
        # Update processes.
        if self.m_cur_pid is not None:
            # The current process is about to terminate.
            if self.m_df_proc.loc[self.m_cur_pid]['life'] < self.m_time - self.m_df_proc.loc[self.m_cur_pid]['birth']:
                self.m_df_proc.loc[self.m_cur_pid, 'state'] = G_PROC_TERM
                self.m_df_proc.loc[self.m_cur_pid, 'state_hist'].append((self.m_time, G_PROC_TERM))
            # Otherwise, switch the current process' state to READY.
            else:
                self.m_df_proc.loc[self.m_cur_pid, 'state'] = G_PROC_READY
                self.m_df_proc.loc[self.m_cur_pid, 'state_hist'].append((self.m_time, G_PROC_READY))

        # Update the current process ID.
        l_ready_proc_ids = list(self.m_df_proc[self.m_df_proc['state'] == G_PROC_READY].index)
        if len(l_ready_proc_ids) > 0:
            cur_pid = np.random.choice(self.m_df_proc[self.m_df_proc['state'] == G_PROC_READY].index)
            self.m_cur_pid = cur_pid
            self.m_df_proc.loc[self.m_cur_pid, 'state'] = G_PROC_RUN
            self.m_df_proc.loc[self.m_cur_pid, 'state_hist'].append((self.m_time, G_PROC_RUN))

    def init_page_data(self):
        l_page = []
        for pid in self.m_df_proc.index:
            l_page_ids = self.m_df_proc.loc[pid]['l_page_ids']
            in_mem = False
            R = 0
            M = 0
            l_page += [(page_id, pid, in_mem, R, M) for page_id in l_page_ids]
        self.m_df_page = pd.DataFrame(l_page, columns=['page_id', 'pid', 'in_mem', 'R', 'M'])
        self.m_df_page = self.m_df_page.set_index('page_id')
        logging.debug('[PageProcSim:init_page_data] Initialized %s pages.' % len(self.m_df_page))

    def gen_page_ref(self):
        cur_proc_data = self.m_df_proc.loc[self.m_cur_pid]
        l_page_ids = cur_proc_data['l_page_ids']
        page_trans_mat = cur_proc_data['page_trans_mat']
        page_ref_hist = cur_proc_data['page_ref_hist']

        if len(page_ref_hist) <= 0:
            ref_page_id = np.random.choice(l_page_ids)
        else:
            last_ref_page_id = page_ref_hist[-1][1]
            last_ref_page_id_idx = l_page_ids.index(last_ref_page_id)
            page_trans_probs = page_trans_mat[last_ref_page_id_idx]
            ref_page_id = np.random.choice(l_page_ids, p=page_trans_probs)

        # TODO
        #   References with modifications can be implemented in addition to read-only references.
        ro = True
        return ref_page_id, ro

    def gen_page_trans_mat_right(self, n_pages):
        """
        Generate a right transition matrix for page references. The sum of each row is 1, but the sum of each column may not be 1.
        :param n_pages:
            int
            > 0
            The number of pages.
        :return:
            ndarray
            A transition matrix.
        """
        trans_mat = np.zeros((n_pages, n_pages))
        for i in range(n_pages):
            trans_mat[i] = np.random.dirichlet([np.abs(n_pages - np.abs(i - k)) for k in range(n_pages)], size=1)[0]
        return trans_mat

    def gen_page_trans_mat_doubly(self, n_pages):
        """
        Generate a doubly transition matrix for page references. The sum of each row is 1, and the sum of each column is 1. s
        :param n_pages:
            int
            > 0
            The number of pages.
        :return:
            ndarray
            A transition matrix.
        """
        trans_mat = np.zeros((n_pages, n_pages))
        for i in range(n_pages):
            full_row = np.random.dirichlet([np.abs(n_pages - np.abs(i - k)) for k in range(n_pages)], size=1)[0]
            full_col = np.random.dirichlet([np.abs(n_pages - np.abs(i - k)) for k in range(n_pages)], size=1)[0]
            row_pre_sum = np.sum(trans_mat[i][:i])
            col_pre_sum = np.sum(trans_mat[:,i][:i])
            row_suc_sum = np.sum(full_row[i:])
            col_suc_sum = np.sum(full_col[i:])
            cal_row_suc = (full_row[i:] / row_suc_sum) * (1 - row_pre_sum)
            cal_col_suc = (full_col[i:] / col_suc_sum) * (1 - col_pre_sum)
            trans_mat[i][i:] = cal_row_suc
            trans_mat[:, i][i:] = cal_col_suc

        return trans_mat


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    ins_sim = PageProcSim(G_N_PROC, G_MIN_N_PAGES_PER_PROC, G_MAX_N_PAGES_PER_PROC, G_MIN_LIFE, G_MAX_LIFE,
                          G_MAX_N_PAGES_IN_MEM)
    ins_sim.start()
