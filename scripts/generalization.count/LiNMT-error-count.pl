#!/usr/bin/perl

use strict;

if(scalar(@ARGV) ne 2) {
  print STDERR "perl error_count.pl src trans\n";
  exit(1);
}

open (SRC, "<", $ARGV[0]) or die "Error: can not read $ARGV[0]!\n";
open (TRA, "<", $ARGV[1]) or die "Error: can not read $ARGV[1]!\n";
open (LOG, ">", "log.txt");

my $line_num = 0;

my $number_lines = 0;
my $time_lines = 0;
my $date_lines = 0;
my $psn_lines = 0;
my $loc_lines = 0;
my $org_lines = 0;

my $err_number_lines = 0;
my $err_time_lines = 0;
my $err_date_lines = 0;
my $err_psn_lines = 0;
my $err_loc_lines = 0;
my $err_org_lines = 0;

while (my $src = <SRC>) {
  my $tra = <TRA>;
  $src =~ s/[\r\n]//g;
  $tra =~ s/[\r\n]//g;
  my @src_fields = split / \|\|\|\| /, $src;
  my @tra_fields = split / \|\|\|\| /, $tra;
  $src = $src_fields[0];
  $tra = $tra_fields[0];
  
  my @src_words = split /\s+/, $src;
  my @tgt_words = split /\s+/, $tra;

  my $src_num_count = 0;
  my $src_time_count = 0;
  my $src_date_count = 0;
  my $src_psn_count = 0;
  my $src_loc_count = 0;
  my $src_org_count = 0;
  foreach my $word (@src_words) {
    if ($word eq "\$number") {
      ++$src_num_count;
    }
    if ($word eq "\$time") {
      ++$src_time_count;
    }
    if ($word eq "\$date") {
      ++$src_date_count;
    }
    if ($word eq "\$psn") {
      ++$src_psn_count;
    }
    if ($word eq "\$loc") {
      ++$src_loc_count;
    }
    if ($word eq "\$org") {
      ++$src_org_count;
    }
  }
  

  my $tra_num_count = 0;
  my $tra_time_count = 0;
  my $tra_date_count = 0;
  my $tra_psn_count = 0;
  my $tra_loc_count = 0;
  my $tra_org_count = 0;
  foreach my $word (@tgt_words) {
    if ($word eq "\$number")
    {
      ++$tra_num_count;
    }
    if ($word eq "\$time")
    {
      ++$tra_time_count;
    }
    if ($word eq "\$date") {
      ++$tra_date_count;
    }
    if ($word eq "\$psn") {
      ++$tra_psn_count;
    }
    if ($word eq "\$loc") {
      ++$tra_loc_count;
    }
    if ($word eq "\$org") {
      ++$tra_org_count;
    }
  }

  

  if ($src_num_count ne 0) {
    ++$number_lines;
    if ($src_num_count ne $tra_num_count) 
    {
      ++$err_number_lines;
      print LOG "[NUMBER] $src\t$tra\n"
    }
  }
  if ($src_time_count ne 0) {
    ++$time_lines;
    if ($src_time_count ne $tra_time_count) {
      ++$err_time_lines;
      print LOG "[TIME] $src\t$tra\n"
    }
  }

  if ($src_date_count ne 0) {
    ++$date_lines;
    if ($src_date_count ne $tra_date_count) {
      ++$err_date_lines;
      print LOG "[DATE] $src\t$tra\n"
    }
  }

  if ($src_psn_count ne 0) {
    ++$psn_lines;
    if ($src_psn_count ne $tra_psn_count) {
      ++$err_psn_lines;
      print LOG "[PSN] $src\t$tra\n"
    }
  }


  if ($src_loc_count ne 0) {
    ++$loc_lines;
    if ($src_loc_count ne $tra_loc_count) {
      ++$err_loc_lines;
      print LOG "[LOC] $src\t$tra\n"
    }
  }


  if ($src_org_count ne 0) {
    ++$org_lines;
    if ($src_org_count ne $tra_org_count) {
      ++$err_org_lines;
      print LOG "[ORG] $src\t$tra\n"
    }
  }

}


my $percent_num = 0;
if ($number_lines ne 0) {
  $percent_num = $err_number_lines / $number_lines * 100;
}

my $percent_time = 0;
if ($time_lines ne 0) {
  $percent_time = $err_time_lines / $time_lines * 100;
}

my $percent_date = 0;
if ($date_lines ne 0) {
  $percent_date = $err_date_lines / $date_lines * 100;
}

my $percent_psn = 0;
if ($psn_lines ne 0) {
  $percent_psn = $err_psn_lines / $psn_lines * 100;
}

my $percent_loc = 0;
if ($loc_lines ne 0) {
  $percent_loc = $err_loc_lines / $loc_lines * 100;
}

my $percent_org = 0;
if ($org_lines ne 0) {
  $percent_org = $err_org_lines / $org_lines * 100;
}



print STDERR "NUM_LINES=$number_lines ERR_NUM=$err_number_lines ERR_PERC=$percent_num \%\n".
             "TIME_LINES=$time_lines ERR_TIME=$err_time_lines ERR_PERC=$percent_time \%\n".
             "DATE_LINES=$date_lines ERR_DATE=$err_date_lines ERR_PERC=$percent_date \%\n".
             "PSN_LINES=$psn_lines ERR_PSN=$err_psn_lines ERR_PERC=$percent_psn \%\n".
             "LOC_LINES=$loc_lines ERR_LOC=$err_loc_lines ERR_PERC=$percent_loc \%\n".
             "ORG_LINES=$org_lines ERR_ORG=$err_org_lines ERR_PERC=$percent_org \%\n";
print LOG "NUM_LINES=$number_lines ERR_NUM=$err_number_lines ERR_PERC=$percent_num \%\n".
          "TIME_LINES=$time_lines ERR_TIME=$err_time_lines ERR_PERC=$percent_time \%\n".
          "DATE_LINES=$date_lines ERR_DATE=$err_date_lines ERR_PERC=$percent_date \%\n".
          "PSN_LINES=$psn_lines ERR_PSN=$err_psn_lines ERR_PERC=$percent_psn \%\n".
          "LOC_LINES=$loc_lines ERR_LOC=$err_loc_lines ERR_PERC=$percent_loc \%\n".
          "ORG_LINES=$org_lines ERR_ORG=$err_org_lines ERR_PERC=$percent_org \%\n";


close (SRC);
close (TRA);
close (LOG);


