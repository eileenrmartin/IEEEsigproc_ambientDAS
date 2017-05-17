#!/usr/bin/env python
import sys
import subprocess
import os
import time
import re
from collections import defaultdict

CEES_RCF = "cees-rcf.stanford.edu"
CEES_MAZAMA = "cees-mazama"
GET_ID = re.compile("^(\d+)")

class job:
  """A generic class for describing a parallel job"""

  def __init__(self, tag):
    """Initialize a job
      tag -- Each job is given a tag to be identified"""
    self.tag = tag

  def preJob(self):
    """What to do before running a job"""

  def checkJobFinishedCorrectly(self):
    """Check whether a job finished correctly"""
    return True
    
  def returnJobCommand(self):
    """Return the command to run the job"""

  def postJob(self):
    """What to do after a job has run"""

# -----------------------------------------------------------------------------
    
class runParallelJob:
  """A generic base class for running jobs in parallel"""

  def __init__(self, jobsToRun):
    """Initialization of the base class
      jobsToRun -- dictionary with a tag -> job"""
    self.jobsToRun = jobsToRun
    self.jobsRunning = {}
    self.jobsFailed = defaultdict(int)

  def startJob(self, tag):
    """Start a job"""
    print("runParallelJob - startJob must be overwritten")
    sys.exit(-1) # Force this to be overwritten

  def checkJobsRunning(self):
    """Check what jobs are running and return a list of finished job tags""" 
    print("runParallelJob - checkJobsRunning must be overwritten")
    sys.exit(-1) # Force this to be overwritten
    
  def checkJobsFinished(self, jobsFinished):
    """Check to see if the jobs finished correctly"""
    print("runParallelJob - checkJobsFinished must be overwritten")
    sys.exit(-1) # Force this to be overwritten
      
  def allJobsFinished(self):
     """What to do when all the jobs are finished"""

  def runJobs(self):
    """Run a series of parallel jobs"""
    print("runParallelJob - runJobs must be overwritten")
    sys.exit(-1) # Force this to be overwritten

# -----------------------------------------------------------------------------

class singleNodeParallel(runParallelJob):
  """Run jobs in parallel on a single node"""

  def __init__(self, jobsToRun):
    runParallelJob.__init__(self, jobsToRun)
    self.processPoll = {}

  def startJob(self, tag, command, stdo, stde):
    if stde:
      efile = open(stde, "w")
      if stdo:
        ofile = open(stdo, "w")
        # Use subprocess.Popen to avoid waiting for command to finish 
        self.processPoll[tag] = subprocess.Popen(command, stderr=efile,
          stdout=ofile, shell=True)
      else:
        self.processPoll[tag] = subprocess.Popen(command, stderr=efile, shell=True)
    else:
      if stdo:
        ofile = open(stdo, "w")
        self.processPoll[tag] = subprocess.Popen(command, stdout=ofile, shell=True)
      else:
        self.processPoll[tag] = subprocess.Popen(command, shell=True)

  def checkJobsRunning(self):
    jobsFinished = []
    for tag, process in self.processPoll.items():
      if process.poll() is not None: # i.e. the job has finished
        jobsFinished.append(tag)
        del self.processPoll[tag] 
    return jobsFinished

  def checkJobsFinished(self, jobsFinished):
    for tag in jobsFinished:
      if not self.jobsRunning[tag].checkJobFinishedCorrectly():
        self.jobsFailed[tag] += 1
        # If the job has failed more than twice give up on it
        if self.jobsFailed[tag] > 2:
          print("Giving up on job", tag)
        else: # Try to run the job again
          self.jobsToRun[tag] = self.jobsRunning[tag]
      del self.jobsRunning[tag]
    
  def runJobs(self, maxJobsRunning, sleepTime=0.5):
    while len(self.jobsToRun) > 0 or len(self.jobsRunning) > 0:
      jobsFinished = self.checkJobsRunning()
      self.checkJobsFinished(jobsFinished)
      if len(self.jobsToRun) > 0 and len(self.jobsRunning) < maxJobsRunning:
        tag, job = self.jobsToRun.popitem()
        print("Starting job", tag)
        self.jobsRunning[tag] = job
        command, stdo, stde = job.returnJobCommand() 
        self.startJob(tag, command, stdo, stde)
      time.sleep(sleepTime)
    self.allJobsFinished()

# -----------------------------------------------------------------------------

class gridParallel(runParallelJob):
  """Run jobs in parallel on a cluster"""

  def __init__(self, jobsToRun, pbsOptions):
    runParallelJob.__init__(self, jobsToRun)
    self.pbs = pbsOptions
    self.processInfo = {} # Contains [jobID, startTime, jobStatus]
    self.swappedQueue = False

  def startJob(self, tag, command, stdo, stde, queue):
    scriptFile = self.pbs.createScript(tag, command, stdo, stde, queue)
    # Submit the job to the cluster
    submitCommand = "qsub %s" % scriptFile
    # " | awk 'match($1, /^[0-9]+/) {print substr($1, RSTART, RLENGTH)}'"
    process = subprocess.run(submitCommand, stdout=subprocess.PIPE, shell=True)
    out = process.stdout.decode("utf-8").rstrip()
    jobID = GET_ID.search(out).group(0)
    startTime = time.time()
    jobStatus = 0
    if process.returncode or not jobID: 
      print("Could not submit job", tag)
      jobStatus = 1
    self.processInfo[tag] = [jobID, startTime, jobStatus]

  def checkJobsRunning(self):
    jobsFinished = []
    for tag, info in list(self.processInfo.items()):
      if info[2]: # not jobStatus == 0, i.e. something went wrong
        jobsFinished.append(tag)
      else:
        qStat = self.pbs.returnJobStatus(info[0])

        if qStat == "C": # Job completed
          print("Finished job", tag)
          jobsFinished.append(tag)

        elif qStat == "R": # Job still running
          # Terminate jobs that are running longer than expected
          runTime = time.time() - info[1]
          if runTime > self.pbs.timeout:
            print("Killing job %s (Bad node)", tag)
            self.pbs.killJob(info[0])
            self.processInfo[tag][2] = 1
            jobsFinished.append(tag)
            
        elif qStat == "H" or qStat == "W" or qStat == "T" or qStat == "S": 
          # Something unusual happened to my job
          print("Killing job %s (Did not run)" % tag)
          self.pbs.killJob(info[0])
          self.processInfo[tag][2] = 1
          jobsFinished.append(tag)

    return jobsFinished

  def runJobs(self, sleepTime):
    while len(self.jobsToRun) > 0 or len(self.jobsRunning) > 0:
      jobsFinished = self.checkJobsRunning()
      self.checkJobsFinished(jobsFinished)

      if len(self.jobsToRun) > 0:
        queue = self.pbs.selectQueue()
        if queue:
          tag, job = self.jobsToRun.popitem()
          print("Starting job", tag)
          self.jobsRunning[tag] = job
          cmd, stdo, stde = job.returnJobCommand()
          self.startJob(tag, cmd, stdo, stde, queue)
      
      elif not self.swappedQueue: # len(self.jobsToRun) == 0 and len(self.jobsRunning) > 0
        newQueue = self.pbs.swapQueues()
        if newQueue: 
          print("Moving remaining jobs to another queue")
          for tag, info in list(self.processInfo.items()):
            qStat = self.pbs.returnJobStatus(info[0])
            if qStat == "Q":
              self.pbs.killJob(info[0])
              self.jobsRunning[tag] = job
              cmd, stdo, stde = job.returnJobCommand()
              self.startJob(tag, cmd, stdo, stde, newQueue)
          self.swappedQueue = True # Only swap queue once

      time.sleep(sleepTime)
    self.allJobsFinished()

  def checkJobsFinished(self, jobsFinished):
    """Check to see if the jobs finished correctly"""
    for tag in jobsFinished:
      if not self.jobsRunning[tag].checkJobFinishedCorrectly() or self.processInfo[tag][2]:
        self.jobsFailed[tag] += 1
        # If the job has failed more than twice give up on it
        if self.jobsFailed[tag] > 2:
          print("Giving up on job", tag)
        else: # Try to run the job again
          self.jobsToRun[tag] = self.jobsRunning[tag]
      del self.jobsRunning[tag]
      del self.processInfo[tag]
    
# -----------------------------------------------------------------------------

class pbs:

  def __init__(self, jobName="youhadonejob", scratchPath="/tmp", timeout=2):
    self.hostname = os.environ["HOSTNAME"]
    self.user = os.environ["USER"]
    self.nodes = 1 # Number of nodes per job
    self.ppn = 16
    if self.hostname == CEES_MAZAMA: 
      self.ppn = 24

    # timeout has to be positive
    if timeout < 1:
      timeout = 1

    # timeout is in hours
    # Convert it to seconds
    # Shorten timeout by 5 minutes, in order to make it shorter than walltime
    self.timeout = timeout * 3600 - 300 

    self.walltime = "%02d:59:59" % (timeout - 1)

    # This list should be ranked from preferred queue to default queue
    self.queue = ["sep"] 
    if timeout <= 2: 
      # Jobs on the default queue have a 2-hour limit
      self.queue.append("default")

    self.scratchPath = scratchPath
    self.jobName = jobName

  def createScript(self, tag, command, stdo, stde, queue): 
    """Create a PBS shell script and return the file name"""
    scriptFile = "%s/sub_%s_%d.sh" % (self.scratchPath, self.jobName, tag)
    f = open(scriptFile,'w')
    script_txt = """
#!/bin/tcsh
#PBS -N %s_%d
#PBS -l nodes=%d:ppn=%d
#PBS -l walltime=%s
#PBS -q %s
#PBS -j oe
#PBS -m a
#PBS -o %s/log_%s_%d.txt 
cd $PBS_O_WORKDIR
%s""" % (self.jobName, tag,
  self.nodes, self.ppn, 
  self.walltime, 
  queue,
  self.scratchPath, self.jobName, tag,
  command)
    if stdo: 
      script_txt += " > " + stdo
    if stde: 
      script_txt += " 2> " + stde
    f.write(script_txt)
    f.close()
    return scriptFile

  def interrogateQueue(self, queue, searchTag, status):
    """Check how many jobs are in this queue, and match the searchTag and status""" 
    cmdTemplate = "qstat %s | grep %s | grep \' %s \' | wc -l" 
    cmd = cmdTemplate % (queue, searchTag, status)
    process = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
    out = process.stdout.decode("utf-8").rstrip()
    if process.returncode:
      print("Problem with qstat")
    return int(out)

  def selectQueue(self):
    """Returns the first queue with less than 2 jobs lines up
       Returns an empty string if none of the queues are free"""
    selectedQueue = ""
    for queue in self.queue:
      if not selectedQueue: 
        nQueued = self.interrogateQueue(queue, self.user, "Q")
        if nQueued < 2: 
          selectedQueue = queue
    return selectedQueue

  def swapQueues(self):
    """Checks whether it would be worth moving the remaining queued jobs
       to another queue. 
       If yes, returns the new queue name. 
       If not, returns an empty string""" 
    newQueue = ""
    # Check if I have queued jobs
    nQueuedTotal = self.interrogateQueue("", self.user, "Q")

    if nQueuedTotal > 0: 
      # Check if one of the queues is empty
      for queue in self.queue:
        if not newQueue: 
          nQueued = self.interrogateQueue(queue, self.user, "Q")
          if nQueued == 0:
            newQueue = queue
    return newQueue

  def returnJobStatus(self, jobID):
    """Return the status of the job in the qstat command""" 
    cmdTemplate = "qstat | grep %s | awk \' { print $5 } \' " 
    cmd = cmdTemplate % (jobID)
    process = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
    out = process.stdout.decode("utf-8").rstrip()
    if process.returncode:
      print("Problem with qstat")
    return out

  def killJob(self, jobID):
    """Kill the job""" 
    cmd = "qdel %s" % jobID
    process = subprocess.run(cmd, shell=True)
    if process.returncode:
      print("Could not kill the job")

# -----------------------------------------------------------------------------
#                 DEFINE THE SPECIFICS OF THE JOB BELOW
# -----------------------------------------------------------------------------

class dasTraining(job):
  def __init__(self, tag):
    job.__init__(self, tag)

  def returnJobCommand(self): 
    # Returns command, standard out, standard error
    return "make trainingPrep%s"%(self.tag + 1), "/scratch/fantine/das/log.%d"%(self.tag + 1), None

# -----------------------------------------------------------------------------

class dasTrainingParallel(gridParallel):
  def __init__(self, njobs):
    jobs = {}
    self.njobs = njobs
    for i in range(njobs):
      jobs[i] = dasTraining(i)
    pbsOptions = pbs(jobName="training", scratchPath="/scratch/fantine/das/", timeout=40)
    gridParallel.__init__(self, jobs, pbsOptions)

  def allJobsFinished(self):
    cmd = "make training"
    process = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
    print('done')

# -----------------------------------------------------------------------------

x = dasTrainingParallel(7)
x.runJobs(1.0)
